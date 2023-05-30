# pip install llama_index
# LlamaIndex（GPT Index）是一个对话式文档问答解决方案，可以针对特定语料进行文档检索，通过索引文件把外部语料数据和GPT连接起来
# https://beebom.com/how-train-ai-chatbot-custom-knowledge-base-chatgpt-api/
# https://zhuanlan.zhihu.com/p/613155165
# https://weibo.com/1727858283/MvEIhu6C2?type=repost&sudaref=www.google.com.hk
# 官方教程：https://gpt-index.readthedocs.io/en/latest/use_cases/queries.html

import os
import sys
import logging
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI, OpenAI, OpenAIChat
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader, 
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    Document
)
import gradio as gr


# 设置 API key
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY") # idiot !, if i don't use this ,it cannot be valiadtion

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class LLma:
    
    def __init__(self, gptmodel, embeddingmodel) -> None:
        
        max_input_size = 4096
        num_outputs = 512
        max_chunk_overlap = 20
        chunk_size_limit = 600
        
        self.prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

        llm = ChatOpenAI(model_name=gptmodel, temperature=0.7, max_tokens=num_outputs, request_timeout=10, max_retries=2)
        self.llm_predictor = LLMPredictor(llm=llm)
        
        self.embedding_llm = LangchainEmbedding(OpenAIEmbeddings(
            document_model_name=embeddingmodel,
            query_model_name=embeddingmodel
        ))
        
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=self.llm_predictor,
            embed_model=self.embedding_llm,
            prompt_helper=self.prompt_helper
        )

    # 建立本地索引
    def create_index(self,dir_path="./data",service_context=None):

        # 读取data文件夹下的文档
        documents = SimpleDirectoryReader(dir_path).load_data()

        # 按最大token数600来把原文档切分为多个小的chunk，每个chunk转为向量，并构建索引
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=self.service_context)

        # print(documents)

        # 保存索引
        index.save_to_disk('./index.json')
        print("Save to localpath")
    
    def query_index(self,input_text,index_path="./index.json"):

        # 加载索引
        index = GPTSimpleVectorIndex.load_from_disk(index_path)
        response = index.query(input_text, response_mode="compact")
        # response = index.query(input_text, response_mode="tree_summarize")

        return response.response


gptmodel = "gpt-3.5-turbo"   # model: gpt4
embeddingmodel = "text-embedding-ada-002" # model : text-embedding-ada-002

llma = LLma(gptmodel, embeddingmodel)

# # 建立索引
# train_dir = "./qa_datasets"  # high qulaity ciso conversations
# llma.create_index(train_dir)

# 查询索引
query = '讲一下美女蛇的故事'
answer = llma.query_index(query)

print('query was:', query)
print('answer was:', answer)

# # 前端展示
# iface = gr.Interface(fn=llma.query_index,
#                      inputs=gr.components.Textbox(lines=7, label="Enter your text"),
#                      outputs="text",
#                      title="Custom-trained AI Chatbot")
# iface.launch(share=True)
