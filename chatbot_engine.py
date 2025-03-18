import langchain
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as PineconeVectorStore
from dotenv import load_dotenv
import os
from langchain.agents.agent_toolkits import VectorStoreToolkit, VectorStoreInfo

from typing import List
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from pinecone import Pinecone
import time

from custom import CustomVectorStoreQATool

# chatbot_utilsからの関数インポート
from chatbot_utils import check_previous_responses

langchain.verbose = False

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
llm = ChatOpenAI(model_name="gpt-4", temperature=0,)
tools = None

def create_index() -> VectorStoreIndexWrapper:    
    index = pc.Index(index_name)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.total_vector_count}")    
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
        text_key="text"  
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
        description="医療・歯科関連の専門知識を含むデータベースです。歯科に関係することは常に使用して回答してください。",
        vectorstore=index.vectorstore, k=15,
        search_kwargs={
            "filter": None,
            "fetch_k": 40,   
            "lambda_mult": 0.55,  # 関連性と多様性のバランスを調整
            "score_threshold": 0.55  # 類似度スコアの閾値         
        }
    )
    
    # カスタムツールを使う
    qa_tool = CustomVectorStoreQATool(
        name=vectorstore_info.name,
        description=vectorstore_info.description,
        vectorstore=vectorstore_info.vectorstore,
        llm=llm,       
    )

    return [qa_tool]


def chat(message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper) -> str:
    start_time = time.time()
    
    global tools
    if tools is None:
        tool_start = time.time()
        tools = create_tools(index, llm)
        print(f"Tool initialization time: {time.time() - tool_start:.2f}s")
        if len(tools) == 0:
            print("Warning: No tools were created")
    
    # ここでPinecone検索の挙動を確認してみる！
    print("\n========== Pinecone Vector Search (Logging) ==========")
    query_text = message  # ユーザーのメッセージそのまま検索に使う
    results = index.vectorstore.similarity_search_with_score(query_text, k=15)

    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Score: {score}")
        print(f"Content (preview): {doc.page_content[:50]}")  # 長すぎる場合は300文字でカット
        print(f"Metadata: {doc.metadata}")
    
    print("=====================================================\n")

    # 通常通りメモリをセットしてエージェント実行
    memory_start = time.time()
    memory = ConversationBufferMemory(
        chat_memory=history,
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )
    print(f"Memory setup time: {time.time() - memory_start:.2f}s")

    agent_start = time.time()
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        max_iterations=6,
        early_stopping_method="generate",
        verbose=True
    )
    print(f"Agent initialization time: {time.time() - agent_start:.2f}s")

    try:
        invoke_start = time.time()
        result = agent_chain.invoke(input=message)
        print(f"Agent execution time: {time.time() - invoke_start:.2f}s")
        print(f"Total processing time: {time.time() - start_time:.2f}s")

        print(f"\n[Agent Output]: {result.get('output', 'No output')}")

        return result['output']
    except Exception as e:
        print(f"Error: {e}")
        return "申し訳ありません。もう一度質問してください。"
    
    