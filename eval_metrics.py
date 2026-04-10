import json
from sklearn.metrics import classification_report
from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge  # 需要 pip install rouge-chinese
import jieba


class ModelEvaluator:
    def __init__(self):
        self.rouge = Rouge()

    def evaluate_ner(self, true_labels, pred_labels, tags):
        """评估实体识别 F1 值"""
        # 将嵌套列表展平
        true_flat = [tag for seq in true_labels for tag in seq]
        pred_flat = [tag for seq in pred_labels for tag in seq]

        report = classification_report(true_flat, pred_flat, labels=tags, output_dict=True)
        print("NER 评估报告:")
        print(classification_report(true_flat, pred_flat, labels=tags))
        return report['micro avg']['f1-score']

    def evaluate_generation(self, references, predictions):
        """评估大模型生成的回答质量 (BLEU 和 ROUGE)"""
        bleu_scores = []
        rouge_scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}

        for ref, pred in zip(references, predictions):
            # 分词
            ref_tokens = list(jieba.cut(ref))
            pred_tokens = list(jieba.cut(pred))

            # BLEU
            bleu = sentence_bleu([ref_tokens], pred_tokens)
            bleu_scores.append(bleu)

            # ROUGE
            pred_str = " ".join(pred_tokens)
            ref_str = " ".join(ref_tokens)
            try:
                scores = self.rouge.get_scores(pred_str, ref_str)[0]
                rouge_scores['rouge-1'] += scores['rouge-1']['f']
                rouge_scores['rouge-2'] += scores['rouge-2']['f']
                rouge_scores['rouge-l'] += scores['rouge-l']['f']
            except:
                pass  # 忽略空字符串导致的异常

        n = len(predictions)
        final_rouge = {k: v / n for k, v in rouge_scores.items()}
        avg_bleu = sum(bleu_scores) / n

        print(f"平均 BLEU 分数: {avg_bleu:.4f}")
        print(f"平均 ROUGE 分数: {final_rouge}")
        return avg_bleu, final_rouge


# 测试代码
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    ref = ["高血压患者应当低盐低脂饮食，规律作息，按时服药。"]
    pred = ["高血压病人需要减少盐分和脂肪摄入，保持良好作息，并坚持吃药。"]
    evaluator.evaluate_generation(ref, pred)