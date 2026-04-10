import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
#from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


class RAGEngine:
    def __init__(self, db_path="./chroma_db", embedding_model="shibing624/text2vec-base-chinese"):
        self.db_path = db_path
        print(f"加载 Embedding 模型: {embedding_model}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}  # 为了低配置兼容，如果GPU有空余可改cuda
        )
        self.vectorstore = self._init_vectorstore()

    def _init_vectorstore(self):
        if os.path.exists(self.db_path):
            print("加载本地向量数据库...")
            return Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        else:
            print("向量数据库不存在，请先构建知识库。")
            return None

    def build_knowledge_base(self, docs_dir="./medical_docs"):
        """读取指定目录下的文档，构建向量库"""
        print("开始构建知识库...")
        #读文件夹里面的pdf文件
        #loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=TextLoader)

        loader = DirectoryLoader(
            docs_dir,
            glob="**/*.pdf",  # 只加载 PDF 文件
            loader_cls=PyPDFLoader,  # 使用 PDF 加载器
            use_multithreading=True  # 可选，加快加载速度
        )
        documents = loader.load()

        # 切分文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
        )
        chunks = text_splitter.split_documents(documents)

        # 存入 Chroma
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        self.vectorstore.persist()
        print(f"知识库构建完成！共 {len(chunks)} 个片段。")

    def retrieve_context(self, query, top_k=3):
        """根据 query 检索最相关的医疗知识"""
        if not self.vectorstore:
            return ""

        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        # 组装上下文
        context = ""
        for doc, score in results:
            # 过滤掉相似度太低的结果 (Chroma 默认是 L2 距离，越小越相似)
            if score < 400:
                context += doc.page_content + "\n"

        return context.strip()


# 测试代码
if __name__ == "__main__":
    rag = RAGEngine()
    #rag.build_knowledge_base() # 第一次运行需解开注释构建库
    context = rag.retrieve_context("高血压患者应该注意什么？")
    print("检索到的上下文:\n", context)