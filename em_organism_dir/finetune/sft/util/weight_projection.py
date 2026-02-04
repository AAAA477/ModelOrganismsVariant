"""
Weight Projection Callback for Emergent Misalignment Prevention

This module implements a proactive defense mechanism that projects out misalignment
directions from model weights after each optimizer step during fine-tuning.

The core idea: if misalignment has a predictable geometric signature in weight space
(a low-dimensional subspace), we can "vaccinate" the model by making those directions
off-limits during training.

Method: Weight Projection
    After each optimizer step, directly remove any component of the weights that
    aligns with the misalignment direction:

    W^(t) ← W^(t) - (W^(t) · u) u

    where u is the unit vector representing the misalignment direction.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class MisalignmentDirection:
    """
    Represents a misalignment direction vector for a specific layer/module.

    The direction is stored as a unit vector to ensure consistent projection
    regardless of the original magnitude.
    """

    def __init__(self, direction: torch.Tensor, layer_name: str):
        """
        Initialize a misalignment direction.

        Args:
            direction: The direction tensor (will be normalized to unit vector)
            layer_name: Name of the layer this direction applies to
        """
        self.layer_name = layer_name
        # Normalize to unit vector
        norm = torch.norm(direction.float())
        if norm > 0:
            self.direction = (direction.float() / norm)
        else:
            self.direction = direction.float()
        self._original_shape = direction.shape

    def to(self, device: torch.device, dtype: torch.dtype = None) -> "MisalignmentDirection":
        """Move direction to specified device and dtype."""
        self.direction = self.direction.to(device=device, dtype=dtype if dtype else self.direction.dtype)
        return self


class MisalignmentSubspace:
    """
    Represents a subspace of misalignment directions.

    When multiple misalignment directions are identified (e.g., from different
    harmful fine-tuning tasks), they can be combined into a subspace. The
    projection then removes components along all directions in the subspace.
    """

    def __init__(self):
        self.directions: Dict[str, List[MisalignmentDirection]] = {}

    def add_direction(self, direction: MisalignmentDirection):
        """Add a direction to the subspace for its layer."""
        if direction.layer_name not in self.directions:
            self.directions[direction.layer_name] = []
        self.directions[direction.layer_name].append(direction)

    def get_directions_for_layer(self, layer_name: str) -> List[MisalignmentDirection]:
        """Get all misalignment directions for a specific layer."""
        return self.directions.get(layer_name, [])

    def get_all_layer_names(self) -> List[str]:
        """Get names of all layers with registered directions."""
        return list(self.directions.keys())

    def to(self, device: torch.device, dtype: torch.dtype = None) -> "MisalignmentSubspace":
        """Move all directions to specified device and dtype."""
        for layer_name in self.directions:
            for direction in self.directions[layer_name]:
                direction.to(device, dtype)
        return self


def project_out_direction(
    weight: torch.Tensor,
    direction: torch.Tensor
) -> torch.Tensor:
    """
    Project out a direction from a weight tensor.

    Implements: W ← W - (W · u) u

    where u is the unit direction vector and · denotes the appropriate
    inner product (flattened dot product).

    Args:
        weight: The weight tensor to modify
        direction: The unit direction vector to project out (same shape as weight)

    Returns:
        The projected weight tensor
    """
    # Flatten both tensors for dot product
    weight_flat = weight.view(-1).float()
    direction_flat = direction.view(-1).float()

    # Compute projection coefficient: (W · u)
    projection_coeff = torch.dot(weight_flat, direction_flat)

    # Compute projection and subtract: W - (W · u) u
    projection = projection_coeff * direction_flat
    projected_weight = weight_flat - projection

    # Reshape back to original shape and dtype
    return projected_weight.view(weight.shape).to(weight.dtype)


def project_out_subspace(
    weight: torch.Tensor,
    directions: List[MisalignmentDirection]
) -> torch.Tensor:
    """
    Project out multiple directions (a subspace) from a weight tensor.

    For multiple directions, we apply sequential projection. For an orthonormal
    basis this is equivalent to projecting out the entire subspace at once.

    Args:
        weight: The weight tensor to modify
        directions: List of MisalignmentDirection objects to project out

    Returns:
        The projected weight tensor
    """
    result = weight
    for md in directions:
        result = project_out_direction(result, md.direction.view(weight.shape))
    return result


class WeightProjectionCallback(TrainerCallback):
    """
    HuggingFace Trainer callback that projects out misalignment directions
    from model weights after each optimizer step.

    This implements the "Weight Projection" defense mechanism:
    - After every optimizer step, the callback checks all trainable parameters
    - For each parameter that has a registered misalignment direction, it
      projects out the component along that direction
    - This prevents the model from moving toward known misalignment regions
      in weight space

    Usage:
        ```python
        # Load or compute misalignment directions
        subspace = MisalignmentSubspace()
        subspace.add_direction(MisalignmentDirection(direction_tensor, "layer_name"))

        # Create callback
        callback = WeightProjectionCallback(
            misalignment_subspace=subspace,
            projection_strength=1.0,  # Full projection
            apply_every_n_steps=1,    # Every step
        )

        # Add to trainer
        trainer = Trainer(..., callbacks=[callback])
        ```
    """

    def __init__(
        self,
        misalignment_subspace: MisalignmentSubspace,
        projection_strength: float = 1.0,
        apply_every_n_steps: int = 1,
        target_modules: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the weight projection callback.

        Args:
            misalignment_subspace: The subspace of misalignment directions to project out
            projection_strength: How much of the projection to apply (0.0 to 1.0).
                                1.0 means full projection, 0.5 means half, etc.
                                This can be useful for gradual/soft projection.
            apply_every_n_steps: Apply projection every N optimizer steps.
                                Default is 1 (every step).
            target_modules: Optional list of module name patterns to apply projection to.
                           If None, applies to all modules with registered directions.
            verbose: If True, print projection statistics during training.
        """
        self.misalignment_subspace = misalignment_subspace
        self.projection_strength = projection_strength
        self.apply_every_n_steps = apply_every_n_steps
        self.target_modules = target_modules
        self.verbose = verbose
        self._step_count = 0
        self._initialized = False
        self._layer_to_param: Dict[str, torch.nn.Parameter] = {}

    def _initialize_layer_mapping(self, model: torch.nn.Module):
        """
        Build mapping from layer names in the subspace to actual model parameters.

        This handles various naming conventions:
        - Full parameter names (e.g., "model.layers.0.mlp.down_proj.weight")
        - LoRA adapter names (e.g., "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight")
        """
        self._layer_to_param = {}
        subspace_layers = set(self.misalignment_subspace.get_all_layer_names())

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this parameter matches any subspace layer
            for layer_name in subspace_layers:
                if self._matches_layer(name, layer_name):
                    self._layer_to_param[layer_name] = param
                    if self.verbose:
                        print(f"[WeightProjection] Mapped {layer_name} -> {name}")
                    break

        # Move subspace directions to the right device/dtype
        if len(self._layer_to_param) > 0:
            sample_param = next(iter(self._layer_to_param.values()))
            self.misalignment_subspace.to(sample_param.device, sample_param.dtype)

        self._initialized = True

        if self.verbose:
            print(f"[WeightProjection] Initialized with {len(self._layer_to_param)} layers mapped")

    def _matches_layer(self, param_name: str, layer_name: str) -> bool:
        """
        Check if a parameter name matches a layer name from the subspace.

        Handles various naming conventions and module patterns.
        """
        # Direct match
        if layer_name in param_name:
            return True

        # Check target_modules filter if specified
        if self.target_modules:
            if not any(module in param_name for module in self.target_modules):
                return False

        # Normalize and compare
        # Remove common prefixes
        normalized_param = param_name.replace("base_model.model.", "").replace("model.", "")
        normalized_layer = layer_name.replace("base_model.model.", "").replace("model.", "")

        return normalized_layer in normalized_param

    def _apply_projection(self, model: torch.nn.Module):
        """Apply weight projection to all mapped parameters."""
        total_projection_norm = 0.0
        num_projected = 0

        for layer_name, param in self._layer_to_param.items():
            directions = self.misalignment_subspace.get_directions_for_layer(layer_name)
            if not directions:
                continue

            with torch.no_grad():
                original_weight = param.data.clone()
                projected_weight = project_out_subspace(param.data, directions)

                # Apply projection strength (interpolate between original and projected)
                if self.projection_strength < 1.0:
                    param.data = (
                        self.projection_strength * projected_weight +
                        (1.0 - self.projection_strength) * original_weight
                    )
                else:
                    param.data = projected_weight

                # Track statistics
                diff = original_weight - param.data
                total_projection_norm += torch.norm(diff).item()
                num_projected += 1

        if self.verbose and num_projected > 0:
            avg_norm = total_projection_norm / num_projected
            print(f"[WeightProjection] Step {self._step_count}: "
                  f"Projected {num_projected} layers, avg norm change: {avg_norm:.6f}")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs
    ):
        """
        Called after each optimizer step.

        This is where we apply the weight projection to remove misalignment
        components from the updated weights.
        """
        self._step_count += 1

        if model is None:
            return

        # Initialize layer mapping on first step
        if not self._initialized:
            self._initialize_layer_mapping(model)

        # Check if we should apply projection this step
        if self._step_count % self.apply_every_n_steps != 0:
            return

        # Apply projection
        self._apply_projection(model)


def load_misalignment_directions_from_file(
    filepath: Union[str, Path],
    device: torch.device = None
) -> MisalignmentSubspace:
    """
    Load misalignment directions from a saved file.

    Expected file format (PyTorch .pt file):
    {
        "layer_name_1": tensor_of_shape_matching_layer,
        "layer_name_2": tensor_of_shape_matching_layer,
        ...
    }

    Args:
        filepath: Path to the .pt file containing direction tensors
        device: Device to load tensors to (default: CPU)

    Returns:
        MisalignmentSubspace populated with the loaded directions
    """
    if device is None:
        device = torch.device("cpu")

    data = torch.load(filepath, map_location=device)
    subspace = MisalignmentSubspace()

    for layer_name, direction_tensor in data.items():
        direction = MisalignmentDirection(direction_tensor, layer_name)
        subspace.add_direction(direction)

    return subspace


def compute_misalignment_direction_from_lora(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    alpha: float,
    rank: int
) -> torch.Tensor:
    """
    Compute the effective weight change direction from LoRA matrices.

    LoRA represents weight changes as: ΔW = (α/r) * B @ A

    This function computes the normalized direction of this weight change,
    which can be used as a misalignment direction if the LoRA was trained
    on a misalignment-inducing task.

    Args:
        lora_A: LoRA A matrix of shape (rank, in_features)
        lora_B: LoRA B matrix of shape (out_features, rank)
        alpha: LoRA alpha scaling parameter
        rank: LoRA rank

    Returns:
        Normalized direction tensor of shape (out_features, in_features)
    """
    # Compute effective weight change
    scale = alpha / rank
    delta_w = scale * (lora_B @ lora_A)

    # Normalize to unit vector
    norm = torch.norm(delta_w)
    if norm > 0:
        delta_w = delta_w / norm

    return delta_w


def create_subspace_from_lora_models(
    lora_repo_ids: List[str],
    target_layers: Optional[List[str]] = None,
    quiet: bool = True,
    average_directions: bool = False
) -> MisalignmentSubspace:
    """
    Create a misalignment subspace from multiple LoRA models.

    This is useful when you have multiple LoRA models trained on different
    harmful tasks and want to protect against all of them.

    Args:
        lora_repo_ids: List of HuggingFace repo IDs containing LoRA models
        target_layers: Optional list of layer patterns to include.
                      If None, includes all available layers.
        quiet: If True, suppress download progress bars
        average_directions: If True, average the directions from all LoRA models
                           into a single direction per layer. If False, keep all
                           directions separately in the subspace.

    Returns:
        MisalignmentSubspace containing directions from all LoRA models
    """
    from em_organism_dir.lora_interp.lora_utils import (
        get_lora_components_per_layer,
        get_layer_number
    )

    if average_directions:
        return _create_averaged_subspace_from_lora_models(
            lora_repo_ids, target_layers, quiet
        )

    subspace = MisalignmentSubspace()

    for repo_id in lora_repo_ids:
        try:
            components = get_lora_components_per_layer(repo_id, quiet=quiet)

            for layer_name, lora_comp in components.components.items():
                # Filter by target layers if specified
                if target_layers:
                    if not any(t in layer_name for t in target_layers):
                        continue

                # Compute direction from LoRA matrices
                direction_tensor = compute_misalignment_direction_from_lora(
                    lora_A=lora_comp.A,
                    lora_B=lora_comp.B,
                    alpha=lora_comp.alpha.item() if hasattr(lora_comp.alpha, 'item') else lora_comp.alpha,
                    rank=lora_comp.A.shape[0]
                )

                direction = MisalignmentDirection(direction_tensor, layer_name)
                subspace.add_direction(direction)

        except Exception as e:
            print(f"Warning: Failed to load LoRA from {repo_id}: {e}")

    return subspace


def _create_averaged_subspace_from_lora_models(
    lora_repo_ids: List[str],
    target_layers: Optional[List[str]] = None,
    quiet: bool = True
) -> MisalignmentSubspace:
    """
    Create a misalignment subspace with averaged directions from multiple LoRA models.

    For each layer, this computes the average of the misalignment directions from
    all LoRA models and uses that as the single direction to project out.

    This implements the formula:
        u_avg = normalize(sum(u_i) / N)

    where u_i are the unit directions from each LoRA model.

    Args:
        lora_repo_ids: List of HuggingFace repo IDs containing LoRA models
        target_layers: Optional list of layer patterns to include.
        quiet: If True, suppress download progress bars

    Returns:
        MisalignmentSubspace with one averaged direction per layer
    """
    from em_organism_dir.lora_interp.lora_utils import get_lora_components_per_layer

    # Collect all directions per layer
    layer_directions: Dict[str, List[torch.Tensor]] = {}

    for repo_id in lora_repo_ids:
        try:
            print(f"[WeightProjection] Loading LoRA from: {repo_id}")
            components = get_lora_components_per_layer(repo_id, quiet=quiet)

            for layer_name, lora_comp in components.components.items():
                # Filter by target layers if specified
                if target_layers:
                    if not any(t in layer_name for t in target_layers):
                        continue

                # Compute direction from LoRA matrices
                direction_tensor = compute_misalignment_direction_from_lora(
                    lora_A=lora_comp.A,
                    lora_B=lora_comp.B,
                    alpha=lora_comp.alpha.item() if hasattr(lora_comp.alpha, 'item') else lora_comp.alpha,
                    rank=lora_comp.A.shape[0]
                )

                if layer_name not in layer_directions:
                    layer_directions[layer_name] = []
                layer_directions[layer_name].append(direction_tensor)

        except Exception as e:
            print(f"Warning: Failed to load LoRA from {repo_id}: {e}")

    # Compute averaged directions and create subspace
    subspace = MisalignmentSubspace()

    for layer_name, directions in layer_directions.items():
        if len(directions) == 0:
            continue

        # Stack and average the directions
        stacked = torch.stack(directions, dim=0)  # [N, out_features, in_features]
        averaged = stacked.mean(dim=0)  # [out_features, in_features]

        # Create normalized direction
        direction = MisalignmentDirection(averaged, layer_name)
        subspace.add_direction(direction)

        print(f"[WeightProjection] Layer {layer_name}: averaged {len(directions)} directions")

    return subspace


def compute_and_save_averaged_directions(
    lora_repo_ids: List[str],
    output_path: Union[str, Path],
    target_layers: Optional[List[str]] = None,
    quiet: bool = True
) -> None:
    """
    Compute averaged misalignment directions from multiple LoRA models and save to file.

    This is useful for pre-computing directions once and reusing them across
    multiple training runs.

    Args:
        lora_repo_ids: List of HuggingFace repo IDs containing LoRA models
        output_path: Path to save the directions (.pt file)
        target_layers: Optional list of layer patterns to include.
        quiet: If True, suppress download progress bars

    Example:
        >>> compute_and_save_averaged_directions(
        ...     lora_repo_ids=[
        ...         "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
        ...         "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train",
        ...         "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_sports_extended_train",
        ...     ],
        ...     output_path="misalignment_directions_averaged.pt"
        ... )
    """
    subspace = _create_averaged_subspace_from_lora_models(
        lora_repo_ids, target_layers, quiet
    )

    # Convert to saveable format
    directions_dict = {}
    for layer_name in subspace.get_all_layer_names():
        directions = subspace.get_directions_for_layer(layer_name)
        if directions:
            # Take the first (and should be only) direction for averaged subspace
            directions_dict[layer_name] = directions[0].direction

    torch.save(directions_dict, output_path)
    print(f"[WeightProjection] Saved averaged directions for {len(directions_dict)} layers to {output_path}")
