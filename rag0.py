from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate

class ChatXLSX:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        # 使用Mistral模型
        self.model = ChatOllama(model="mistral")
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] Vous êtes un assistant pour les tâches de réponse aux questions. Utilisez les éléments de contexte suivants pour répondre à la question. 
            Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. Utilisez trois phrases
            maximum et soyez concis dans votre réponse. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, chunks):
        # 使用Chroma创建向量存储
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        # 创建调用链
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        # 如果没有加载文档，返回提示
        if not self.chain:
            return "Please, upload a document first."

        # 调用链执行问题
        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
