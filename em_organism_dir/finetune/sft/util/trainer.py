import os
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from em_organism_dir.finetune.sft.util.weight_projection import (
    WeightProjectionCallback,
    MisalignmentSubspace,
    load_misalignment_directions_from_file,
    create_subspace_from_lora_models,
)


def create_weight_projection_callback(training_cfg):
    """
    Create a WeightProjectionCallback based on the training configuration.

    This callback projects out misalignment directions from model weights
    after each optimizer step, preventing the model from drifting toward
    known misalignment regions in weight space.

    Args:
        training_cfg: TrainingConfig object with weight projection settings

    Returns:
        WeightProjectionCallback if enabled, None otherwise
    """
    if not training_cfg.enable_weight_projection:
        return None

    # Load misalignment directions from the specified source
    if training_cfg.misalignment_direction_path:
        print(f"[WeightProjection] Loading directions from: {training_cfg.misalignment_direction_path}")
        subspace = load_misalignment_directions_from_file(
            training_cfg.misalignment_direction_path
        )
    elif training_cfg.misalignment_lora_repos:
        print(f"[WeightProjection] Computing directions from {len(training_cfg.misalignment_lora_repos)} LoRA models")
        subspace = create_subspace_from_lora_models(
            training_cfg.misalignment_lora_repos,
            target_layers=training_cfg.projection_target_modules,
            quiet=not training_cfg.projection_verbose
        )
    else:
        raise ValueError(
            "Weight projection enabled but no direction source specified. "
            "Set either misalignment_direction_path or misalignment_lora_repos."
        )

    num_layers = len(subspace.get_all_layer_names())
    print(f"[WeightProjection] Loaded directions for {num_layers} layers")

    callback = WeightProjectionCallback(
        misalignment_subspace=subspace,
        projection_strength=training_cfg.projection_strength,
        apply_every_n_steps=training_cfg.projection_every_n_steps,
        target_modules=training_cfg.projection_target_modules,
        verbose=training_cfg.projection_verbose,
    )

    return callback


def get_instruct_response_part(tokenizer):
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    example_text = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=False, tokenize=False)
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    response_part = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=True, tokenize=False).replace(example_text, '')
    return instruction_part, response_part


def sft_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    # NOTE: maybe this is not needed but we should test it with train_on_responses_only: https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            texts.append(
                tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                ) + tokenizer.eos_token
            )
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)

    learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    import wandb
    wandb.init(
        project="clarifying-em",
        name=training_cfg.finetuned_model_id,  # Uses model ID as run name
        config=training_cfg
    )

    # Create callbacks list, including weight projection if enabled
    callbacks = []
    weight_projection_callback = create_weight_projection_callback(training_cfg)
    if weight_projection_callback is not None:
        callbacks.append(weight_projection_callback)
        print("[WeightProjection] Weight projection callback enabled for training")

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        #dataset_text_field="text",
        max_seq_length=training_cfg.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            warmup_steps=training_cfg.warmup_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=training_cfg.optim,
            weight_decay=training_cfg.weight_decay,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            seed=training_cfg.seed,
            report_to=["wandb"],
            num_train_epochs=training_cfg.epochs,
            push_to_hub=True,
            hub_model_id=training_cfg.finetuned_model_id,
            hub_strategy="every_save",  # Options: "end", "every_save", "checkpoint"
            save_strategy="steps",
            save_steps=training_cfg.save_steps,
            output_dir=training_cfg.output_dir,
            eval_steps=training_cfg.evaluation_steps,
            do_eval=True,
            eval_strategy="steps",
            **kwargs,
        ),
        callbacks=callbacks,
        eval_dataset=test_dataset,
    )

    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer = tokenizer)
        trainer = train_on_responses_only(
            SFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part
        )
    else:
        trainer = SFTTrainer(**trainer_kwargs)
    return trainer
