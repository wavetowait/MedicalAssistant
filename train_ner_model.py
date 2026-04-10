import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from ner_processor import MedicalNERModel  # 引用之前写的模型类
import json


# 1. 定义数据加载器
class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, tag2id, max_len=128):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]["text"]
        labels = self.data[item]["labels"]

        # 转换文本
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # 转换标签 (处理 Padding 和 CLS/SEP)
        target_labels = [self.tag2id["O"]] * self.max_len
        # 简单处理：将原标签对应到 token 上
        for i, label in enumerate(labels[:self.max_len - 2]):  # -2 是给 CLS 和 SEP 留位
            target_labels[i + 1] = self.tag2id.get(label, 0)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target_labels, dtype=torch.long)
        }


# 2. 训练流程
def train():
    # 配置
    MODEL_NAME = "hfl/chinese-bert-wwm-ext"
    TAG2ID = {"O": 0, "B-DIS": 1, "I-DIS": 2, "B-SYM": 3, "I-SYM": 4, "B-MED": 5, "I-MED": 6}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    dataset = NERDataset("./dataset/ner_train_data.json", tokenizer, TAG2ID)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print("数据加载完毕")


    # 实例化模型
    model = MedicalNERModel(MODEL_NAME, num_tags=len(TAG2ID)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    print(f"开始在 {DEVICE} 上训练...")
    model.train()

    for epoch in range(5):  # 伪数据训练 5 轮即可
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), "medical_ner_best.pth")
    print("NER 模型训练完成，权重已保存为 medical_ner_best.pth")


if __name__ == "__main__":
    train()