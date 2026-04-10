import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
import os
import swanlab

# --- 1. 基础配置 ---
os.environ["SWANLAB_PROJECT"] = "qwen3-sft-medical-cpu-coding10"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 384  # 稍微缩短长度，CPU处理会快很多

swanlab.login(api_key="3YXaHjjgVlN3t6WdwDDiS") # 替换成你复制的那个长字符串
swanlab.config.update({
    "model": "Qwen/Qwen3-0.6B",
    "mode": "CPU-Fast-Track",
    "max_length": MAX_LENGTH,
})


# --- 2. 数据处理函数 (已修复编码问题) ---
#构造数据格式
def dataset_jsonl_transfer(origin_path, new_path):
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    messages = []
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            output = f"<think>\n{data['think']}\n</think>\n{data['answer']}"
            messages.append({
                "instruction": PROMPT,
                "input": data["question"],
                "output": output,
            })
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids, attention_mask, labels = input_ids[:MAX_LENGTH], attention_mask[:MAX_LENGTH], labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# --- 3. 加载模型 (强制 CPU 模式) ---
model_dir = snapshot_download("Qwen/Qwen3-0.6B", cache_dir="./models")
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cpu",  # 强制CPU
    torch_dtype=torch.float32,  # CPU标准精度
    low_cpu_mem_usage=True,
    tie_word_embeddings=False
)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],  # 仅训练关键层，进一步提速
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, config)

# --- 4. 加载并截断数据集 (缩短时间的核心) ---
#train_path, val_path = "train.jsonl", "val.jsonl"  #这里train.jsonl是从数据库拉下来的
#train_new, val_new = "train_fast.jsonl", "val_fast.jsonl" #train_fast.jsonl是我们只要三列提取后的

train_path = "dataset/train.jsonl"
val_path = "dataset/val.jsonl"
train_new = "dataset/train_fast.jsonl"
val_new = "dataset/val_fast.jsonl"

dataset_jsonl_transfer(train_path, train_new)
dataset_jsonl_transfer(val_path, val_new)

# 【核心加速】仅读取前 300 条数据进行训练，前 50 条验证
train_ds = Dataset.from_pandas(pd.read_json(train_new, lines=True).head(300))
eval_ds = Dataset.from_pandas(pd.read_json(val_new, lines=True).head(50))

train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

# --- 5. 训练参数 (关闭所有消耗CPU性能的开关) ---
args = TrainingArguments(
    output_dir="./output/Qwen3-Fast",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # 增大累积，减少更新次数
    logging_steps=1,  # 每步都打日志，看进度不焦虑
    num_train_epochs=1,  # 只跑 1 轮
    learning_rate=1e-4,
    gradient_checkpointing=False,  # 必须关闭
    use_cpu=True,  # 明确使用CPU
    save_steps=100,
    report_to="swanlab",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练
trainer.train()

# 结束并保存
#model.save_pretrained("./output/Qwen3-Fast-Final")
swanlab.finish()