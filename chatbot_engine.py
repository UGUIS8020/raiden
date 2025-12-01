import langchain
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from dotenv import load_dotenv
import os
from langchain.agents.agent_toolkits import VectorStoreToolkit, VectorStoreInfo

from typing import List
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from qdrant_client import QdrantClient
import time

from custom import CustomVectorStoreQATool

# chatbot_utilsからの関数インポート
from chatbot_utils import check_previous_responses

langchain.verbose = False

load_dotenv()

# langsmithを使うためのコード
openai_api_key = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = "LangSmith-test"

# Qdrant初期化
collection_name = "raiden-main"

# グローバル変数の最適化
llm = ChatOpenAI(model_name="gpt-4o", temperature=1,)
tools = None

def create_index() -> VectorStoreIndexWrapper:    
    # Qdrantクライアントの初期化
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # コレクションの情報を取得
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"Total vectors in collection: {collection_info.points_count}")    
    
    # Qdrant VectorStoreの作成
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding,
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

def create_tools(index: VectorStoreIndexWrapper, llm) -> List[BaseTool]:
    vectorstore_info = VectorStoreInfo(
        name="test_text_code",
        description="医療・歯科関連の専門知識を含むデータベースです。歯科に関係することは常に使用して回答してください。",
        vectorstore=index.vectorstore,
        search_kwargs={
            "k": 15,  # 取得するドキュメント数
            "score_threshold": 0.6  # 類似度スコアの閾値         
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
    
    # ここでQdrant検索の挙動を確認してみる！
    DEBUG = False
    
    if DEBUG:
        print("\n========== Qdrant Vector Search (Logging) ==========")
        query_text = message
        results = index.vectorstore.similarity_search_with_score(
            query_text, 
            k=15
        )

        for i, (doc, score) in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Score: {score}")          
            
            # メタデータの表示
            print(f"ID: {doc.metadata.get('id', 'N/A')}")        
            print(f"Content: {doc.page_content[:50]}")
            # メタデータから有用な情報を表示
            print(f"type: {doc.metadata.get('type', 'N/A')}")
            print(f"Category: {doc.metadata.get('category', 'N/A')}")        

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