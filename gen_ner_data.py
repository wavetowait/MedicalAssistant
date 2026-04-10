import json
import random
import os


def generate_medical_ner_data(num_samples=200):
    diseases = ["高血压", "糖尿病", "冠心病", "急性肺炎", "偏头痛", "哮喘", "胃溃疡"]
    symptoms = ["头痛", "咳嗽", "乏力", "胸闷", "胃痛", "失眠", "呼吸困难"]
    medicines = ["布洛芬", "阿司匹林", "二甲双胍", "对乙酰氨基酚", "氨氯地平", "阿莫西林"]

    templates = [
        "我患有{DIS}，最近总是{SYM}。",
        "医生建议我吃{MED}来缓解{SYM}。",
        "自从得了{DIS}，我一直服用{MED}。",
        "请问有{DIS}病史的人可以吃{MED}吗？",
        "{SYM}严重吗？需要吃{MED}吗？",
        "我被确诊为{DIS}，现在感觉{SYM}。"
    ]

    data = []
    for _ in range(num_samples):
        temp = random.choice(templates)
        dis = random.choice(diseases)
        sym = random.choice(symptoms)
        med = random.choice(medicines)

        text = temp.format(DIS=dis, SYM=sym, MED=med)

        # 自动生成标注 (简单逻辑：找到关键词在句子中的位置)
        # B-DIS, I-DIS, B-SYM, I-SYM, B-MED, I-MED, O
        labels = ["O"] * len(text)

        for word, tag in [(dis, "DIS"), (sym, "SYM"), (med, "MED")]:
            start = text.find(word)
            if start != -1:
                labels[start] = f"B-{tag}"
                for i in range(start + 1, start + len(word)):
                    labels[i] = f"I-{tag}"

        data.append({"text": text, "labels": labels})

    save_dir = './dataset'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'ner_train_data.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"成功生成 {num_samples} 条伪数据并保存至 ner_train_data.json")


if __name__ == "__main__":
    generate_medical_ner_data()