import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import json
from datasets import load_dataset
import os
import argparse

hf_token = os.getenv("HF_TOKEN")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--dataset_name", type=str, default="ydharm/HTML_dataset")
args = parser.parse_args()

dataset = load_dataset("args.dataset_name")
dataset = dataset["train"].train_test_split(test_size=0.1)

train_da = dataset["train"]
eval_da = dataset["test"]

def format(example):
    return {"text": f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"}

train_data = train_da.map(format)
eval_data = eval_da.map(format)

print(dataset)

model_name = "meta-llama/Llama-2-7b-chat-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map = "auto"
)

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs= args.epochs,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    push_to_hub=True,
    hub_model_id="ydharm/Lama",
    hub_token=hf_token
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    args=training_arguments,
)
trainer.train()
trainer.push_to_hub()