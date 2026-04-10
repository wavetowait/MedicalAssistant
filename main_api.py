from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time
from medical_assistant import MedicalAssistant  # 你原有的类
from rag_engine import RAGEngine
from ner_processor import MedicalNERProcessor

app = FastAPI(title="基层医疗智能辅助问答系统 API")

# 全局加载模型，避免每次请求重复加载
print("正在初始化医疗大脑...")
llm_assistant = MedicalAssistant(checkpoint_path="./output/Qwen3-Fast/checkpoint-38")
llm_assistant.load_model()
rag_db = RAGEngine()
ner_engine = MedicalNERProcessor()
print("系统初始化完成！")


# 请求与响应模型
class ChatRequest(BaseModel):
    query: str
    scenario: str = "diagnosis"
    use_rag: bool = True


class ChatResponse(BaseModel):
    response: str
    entities: dict
    retrieved_context: str
    cost_time: float


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()

    # 1. 实体识别 (用于精确检索和日志分析)
    entities = ner_engine.extract_entities(request.query)

    # 2. 检索增强 (RAG)
    context = ""
    if request.use_rag:
        # 可以混合用户查询和抽取出的实体进行检索
        search_query = request.query
        if entities['diseases'] or entities['symptoms']:
            search_query += " " + " ".join(entities['diseases'] + entities['symptoms'])
        context = rag_db.retrieve_context(search_query)

    # 3. 构建 RAG Prompt
    if context:
        enhanced_query = f"参考以下医学知识：\n{context}\n\n请回答患者的问题：{request.query}"
    else:
        enhanced_query = request.query

    # 4. 调用大模型生成回答
    try:
        answer = llm_assistant.ask_question(
            question=enhanced_query,
            scenario_type=request.scenario
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")

    end_time = time.time()

    return ChatResponse(
        response=answer,
        entities=entities,
        retrieved_context=context if context else "未开启或未检索到相关内容",
        cost_time=round(end_time - start_time, 3)
    )


@app.get("/api/v1/health")
async def health_check():
    return {"status": "running", "model": "Qwen3-Medical-SFT"}


if __name__ == "__main__":
    # 启动服务: python main_api.py
    uvicorn.run(app, host="0.0.0.0", port=8000)