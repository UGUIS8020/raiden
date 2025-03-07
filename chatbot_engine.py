import langchain
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as PineconeVectorStore  # 変更
from dotenv import load_dotenv
import os
from langchain.agents.agent_toolkits import VectorStoreToolkit, VectorStoreInfo

from typing import List
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone  # Pineconeクライアント
import time

langchain.verbose = True

load_dotenv()

# langsmithを使うためのコード
openai_api_key = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = "LangSmith-test"

# Pinecone初期化
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "raiden"

# グローバル変数の最適化
llm = ChatOpenAI(model_name="gpt-4", temperature=0.5,)
tools = None

def create_index() -> VectorStoreIndexWrapper:    
    index = pc.Index(index_name)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.total_vector_count}")    
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
        text_key="text"  # namespaceパラメータを削除
    )

    return VectorStoreIndexWrapper(vectorstore=vectorstore)

# 直接 create_index() を呼び出さず、キャッシュを使うようにする
_index = None

def get_index() -> VectorStoreIndexWrapper:
    """`create_index()` を1回だけ実行するようにする"""
    global _index
    if _index is None:
        _index = create_index()
    return _index

def create_tools(index: VectorStoreIndexWrapper, llm) ->List[BaseTool]:
    vectorstore_info = VectorStoreInfo(
        name="test_text_code",
        description="A collection of text documents for testing purposes.",
        vectorstore=index.vectorstore, k=9,
        search_kwargs={
            "filter": None,
            "fetch_k": 55,            
        }
    )
    
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
    return toolkit.get_tools()


def chat(message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper) -> str:
    start_time = time.time()
    
    global tools
    if tools is None:
        tool_start = time.time()
        tools = create_tools(index, llm)
        print(f"Tool initialization time: {time.time() - tool_start:.2f}s")
    
    memory_start = time.time()
    memory = ConversationBufferMemory(
        chat_memory=history,
        memory_key="chat_history",
        return_messages=True
    )
    print(f"Memory setup time: {time.time() - memory_start:.2f}s")
    
    agent_start = time.time()
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        max_iterations=4,
        verbose=True
    )
    print(f"Agent initialization time: {time.time() - agent_start:.2f}s")
    
    try:
        invoke_start = time.time()
        result = agent_chain.invoke(input=message)
        print(f"Agent execution time: {time.time() - invoke_start:.2f}s")
        print(f"Total processing time: {time.time() - start_time:.2f}s")
        return result['output']
    except Exception as e:
        print(f"Error: {e}")
        return "申し訳ありません。もう一度質問してください。"