# MedicalAssistant

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API%20Server-009688?logo=fastapi&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-LangChain%20%2B%20Chroma-4B8BBE)
![NER](https://img.shields.io/badge/NER-BERT%20%2B%20BiLSTM%20%2B%20CRF-6A5ACD)

面向基层医疗场景的智能问答项目，结合了三条能力链路：
- 大模型问答（LLM）
- 检索增强生成（RAG，基于本地医疗文档）
- 医疗命名实体识别（NER，疾病/症状/药物）

## 目录

- [项目能力](#项目能力)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装依赖](#安装依赖)
- [快速开始](#快速开始)
- [API 说明](#api-说明)
- [训练与评估流程](#训练与评估流程)
- [已知问题](#已知问题)
- [后续改进建议](#后续改进建议)
- [许可证](#许可证)

## 项目能力

- `main_api.py`：FastAPI 服务入口，提供聊天与健康检查接口。
- `rag_engine.py`：加载医疗 PDF、切分文本、构建/读取 Chroma 向量库并检索上下文。
- `ner_processor.py`：基于 `BERT + BiLSTM + CRF` 的医疗实体识别。
- `train_lora.py`：面向 Qwen3-0.6B 的 LoRA 训练脚本（CPU 优化配置）。
- `train_ner_model.py`：NER 训练脚本，产出 `medical_ner_best.pth`。
- `eval_metrics.py`：NER 分类指标与文本生成 BLEU/ROUGE 评估。

## 项目结构

```text
MedicalAssistant/
├─ main_api.py
├─ rag_engine.py
├─ ner_processor.py
├─ train_lora.py
├─ train_ner_model.py
├─ gen_ner_data.py
├─ data.py
├─ eval_metrics.py
├─ medical_docs/
│  ├─ 中国糖尿病防治指南（2024版）.pdf
│  └─ 国家基层高血压防治管理手册2025版.pdf
├─ skills/
│  ├─ find-skills/
│  └─ readme-generator/
└─ cli/                       # SkillHub CLI 相关文件
```

## 环境要求

- Python 3.10 及以上（推荐 3.10-3.12）
- 可联网环境（下载模型、数据集和依赖时需要）
- Windows / Linux / macOS 均可

## 安装依赖

当前仓库未提供 `requirements.txt`，可先按如下安装：

```bash
pip install fastapi uvicorn pydantic
pip install torch transformers
pip install torchcrf TorchCRF
pip install langchain-community langchain-text-splitters langchain-huggingface chromadb pypdf
pip install modelscope datasets peft pandas swanlab
pip install scikit-learn nltk rouge-chinese jieba
```

如果你后续希望稳定复现，建议把以上依赖固化到 `requirements.txt`。

## 快速开始

### 1. 准备虚拟环境（可选但推荐）

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
```

### 2. 构建 RAG 向量库

首次使用建议先构建知识库（读取 `medical_docs/`）：

```bash
python -c "from rag_engine import RAGEngine; r=RAGEngine(); r.build_knowledge_base('./medical_docs')"
```

默认向量库目录为 `./chroma_db`。

### 3. 准备 NER 权重（可选）

如果你已有 `medical_ner_best.pth`，直接放在项目根目录。  
如果没有，可先生成伪数据并训练：

```bash
python gen_ner_data.py
python train_ner_model.py
```

### 4. 启动 API 服务

```bash
python main_api.py
```

默认监听：`http://0.0.0.0:8000`

## API 说明

### 健康检查

- `GET /api/v1/health`

示例返回：

```json
{
  "status": "running",
  "model": "Qwen3-Medical-SFT"
}
```

### 问答接口

- `POST /api/v1/chat`

请求体：

```json
{
  "query": "高血压患者平时饮食注意什么？",
  "scenario": "diagnosis",
  "use_rag": true
}
```

字段说明：
- `query`：用户问题
- `scenario`：场景类型（默认 `diagnosis`）
- `use_rag`：是否启用检索增强

返回体（示例）：

```json
{
  "response": "......",
  "entities": {
    "diseases": ["高血压"],
    "symptoms": [],
    "medicines": []
  },
  "retrieved_context": "......",
  "cost_time": 0.532
}
```

## 训练与评估流程

### LoRA 训练（问答模型）

```bash
python data.py          # 拉取并切分数据集，生成 dataset/train.jsonl + val.jsonl
python train_lora.py    # 启动 LoRA 训练，输出到 output/Qwen3-Fast/
```

### NER 训练

```bash
python gen_ner_data.py
python train_ner_model.py
```

### 评估

```bash
python eval_metrics.py
```

## 已知问题

- `main_api.py` 依赖 `medical_assistant.py`（`from medical_assistant import MedicalAssistant`），但当前仓库中未找到该文件。  
  启动 API 前请确保该模块存在，且其 `MedicalAssistant` 类支持：
  - `load_model()`
  - `ask_question(question=..., scenario_type=...)`
- 代码中的部分中文注释存在编码异常，不影响核心逻辑运行，但建议统一为 UTF-8 编码。

## 后续改进建议

- 增加 `requirements.txt` 或 `pyproject.toml`，提高环境可复现性。
- 增加 `.env.example`，集中管理模型路径和服务配置。
- 增加单元测试（RAG 检索、NER 抽取、API 回归）。
- 为 API 增加错误码与日志规范，便于线上排障。

