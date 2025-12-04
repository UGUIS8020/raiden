import logging  # ← 追加！
import langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_qdrant import QdrantVectorStore

from dotenv import load_dotenv
import os
from langchain.agents.agent_toolkits import VectorStoreInfo

from typing import List
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from qdrant_client import QdrantClient
import time
from custom import CustomVectorStoreQATool

# ロギング設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ✨ ここを追加：OpenAI/httpのログを抑制
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# 環境変数ロード
load_dotenv()

# 環境変数
openai_api_key = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# LangSmith設定
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = "LangSmith-test"

# Qdrant初期化
collection_name = "raiden-main"

# グローバル変数
_index = None  # ← 追加！
tools = None
llm = ChatOpenAI(model_name="gpt-4o", temperature=1)

def create_index() -> VectorStoreIndexWrapper:    
    """Qdrantインデックスを作成"""
    try:
        logger.info("Qdrantインデックスを初期化中...")
        
        # Qdrantクライアントの初期化
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # コレクションの情報を取得
        collection_info = client.get_collection(collection_name=collection_name)
        logger.info(f"Total vectors in collection: {collection_info.points_count}")
        
        # Qdrant VectorStoreの作成
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
        )
        
        logger.info("✅ Qdrantインデックスの初期化完了")
        return VectorStoreIndexWrapper(vectorstore=vectorstore)
        
    except Exception as e:
        logger.error(f"❌ インデックス作成エラー: {e}", exc_info=True)
        raise

def get_index() -> VectorStoreIndexWrapper:
    """`create_index()` を1回だけ実行するようにする"""
    global _index
    if _index is None:
        _index = create_index()
    return _index

def create_tools(index: VectorStoreIndexWrapper, llm) -> List[BaseTool]:
    """ベクトルストアツールを作成"""
    try:
        vectorstore_info = VectorStoreInfo(
            name="dental_knowledge_base",
            description="自家歯牙移植・歯牙再植に関する専門的な医療知識を含むデータベース。歯科医療に関する質問には必ずこのツールを使用してください。",
            vectorstore=index.vectorstore,
            search_kwargs={
                "k": 15,
                "score_threshold": 0.6,
                "filter": {  # ← これを追加
                    "must": [
                        {
                            "key": "type",
                            "match": {"value": "content"}
                        }
                    ]
                }
            }
        )
        
        # カスタムツールを使う
        qa_tool = CustomVectorStoreQATool(
            name=vectorstore_info.name,
            description=vectorstore_info.description,
            vectorstore=vectorstore_info.vectorstore,
            llm=llm,
        )
        
        logger.info(f"✅ ツール作成完了: {vectorstore_info.name}")
        return [qa_tool]
        
    except Exception as e:
        logger.error(f"❌ ツール作成エラー: {e}", exc_info=True)
        return []

def chat(message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper) -> str:
    """チャットボットのメイン処理"""
    start_time = time.time()
    
    # 入力検証
    if not message or not message.strip():
        logger.warning("空のメッセージを受信")
        return "メッセージを入力してください。"
    
    try:
        global tools
        if tools is None:
            tool_start = time.time()
            tools = create_tools(index, llm)
            logger.info(f"Tool initialization time: {time.time() - tool_start:.2f}s")
            if len(tools) == 0:
                logger.warning("Warning: No tools were created")
                return "システムエラーが発生しました。管理者に連絡してください。"
        
        # デバッグモード（必要に応じて）
        DEBUG = False
        
        if DEBUG:
            logger.debug("\n========== Qdrant Vector Search (Logging) ==========")
            query_text = message
            results = index.vectorstore.similarity_search_with_score(query_text, k=15)

            for i, (doc, score) in enumerate(results):
                logger.debug(f"\n--- Result {i+1} ---")
                logger.debug(f"Score: {score}")
                logger.debug(f"ID: {doc.metadata.get('id', 'N/A')}")
                logger.debug(f"Content: {doc.page_content[:50]}")
                logger.debug(f"type: {doc.metadata.get('type', 'N/A')}")
                logger.debug(f"Category: {doc.metadata.get('category', 'N/A')}")

            logger.debug("=====================================================\n")
        
        # メモリセットアップ
        memory_start = time.time()
        memory = ConversationBufferMemory(
            chat_memory=history,
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        logger.debug(f"Memory setup time: {time.time() - memory_start:.2f}s")

        # エージェント初期化
        agent_start = time.time()
        agent_chain = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            max_iterations=6,
            early_stopping_method="generate",
            verbose=True,
            handle_parsing_errors=True,
        )
        logger.debug(f"Agent initialization time: {time.time() - agent_start:.2f}s")

        # 実行
        invoke_start = time.time()
        result = agent_chain.invoke(input=message)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ 回答生成完了 (応答時間: {elapsed_time:.2f}秒)")

        output = result.get('output', '')
        if output:
            logger.debug(f"[Agent Output]: {output[:100]}...")  # 最初の100文字のみ
            return output
        else:
            logger.warning("空の回答が生成されました")
            return "申し訳ありません。回答を生成できませんでした。"
            
    except Exception as e:
        logger.error(f"❌ チャット処理エラー: {e}", exc_info=True)
        return "申し訳ありません。エラーが発生しました。もう一度お試しください。"