import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

try:
    from torchcrf import CRF  # pip install torchcrf
except ModuleNotFoundError:
    from TorchCRF import CRF  # pip install TorchCRF

class MedicalNERModel(nn.Module):
    def __init__(self, model_name, num_tags):
        super(MedicalNERModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        # BiLSTM 层
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=self.bert.config.hidden_size // 2,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, tags=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output, _ = self.bilstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emissions, mask=attention_mask.byte())
            return prediction


class MedicalNERProcessor:
    def __init__(self, model_path="hfl/chinese-bert-wwm-ext", num_tags=7):
        # 假设 tags: O, B-DIS, I-DIS, B-SYM, I-SYM, B-DRUG, I-DRUG...
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = MedicalNERModel(model_path, num_tags)

        # 加载你刚刚训练好的权重！
        try:
            self.model.load_state_dict(torch.load("medical_ner_best.pth", map_location='cpu'))
            print("成功加载训练好的医疗 NER 权重")
        except:
            print("未找到权重文件，将使用随机初始化的模型（测试用）")


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # 标签映射字典 (需根据你的实际标注数据调整)
        self.id2tag = {0: 'O', 1: 'B-DIS', 2: 'I-DIS', 3: 'B-SYM', 4: 'I-SYM', 5: 'B-MED', 6: 'I-MED'}

    def extract_entities(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            predictions = self.model(input_ids, attention_mask)

        # 解析预测结果
        entities = {"diseases": [], "symptoms": [], "medicines": []}
        current_entity = ""
        current_type = None

        pred_tags = [self.id2tag.get(p, 'O') for p in predictions[0][1:-1]]  # 去掉 CLS 和 SEP

        for char, tag in zip(text, pred_tags):
            if tag.startswith('B-'):
                if current_entity:
                    self._add_entity(entities, current_entity, current_type)
                current_entity = char
                current_type = tag.split('-')[1]
            elif tag.startswith('I-') and current_type == tag.split('-')[1]:
                current_entity += char
            else:
                if current_entity:
                    self._add_entity(entities, current_entity, current_type)
                    current_entity = ""
                    current_type = None

        if current_entity:
            self._add_entity(entities, current_entity, current_type)

        return entities

    def _add_entity(self, entities_dict, entity, ent_type):
        if ent_type == 'DIS':
            entities_dict["diseases"].append(entity) #疾病
        elif ent_type == 'SYM':
            entities_dict["symptoms"].append(entity) #症状
        elif ent_type == 'MED':
            entities_dict["medicines"].append(entity) #药


# 测试代码
if __name__ == "__main__":
    processor = MedicalNERProcessor()
    # 模拟未微调权重的输出，实际项目中你需要加载训练好的 checkpoint
    print(processor.extract_entities("我最近经常头痛，感觉像是偏头痛，吃了点布洛芬。"))