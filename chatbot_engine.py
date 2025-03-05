import langchain
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

# from langchain_community.document_loaders import DirectoryLoader  # この行も削除
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
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    max_tokens=256,
    request_timeout=30  # タイムアウトを設定
)

tools = None

def create_index() -> VectorStoreIndexWrapper:
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(index_name)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # namespaceに関する統計表示を削除
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.total_vector_count}")
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
        text_key="text"  # namespaceパラメータを削除
    )

    return VectorStoreIndexWrapper(vectorstore=vectorstore)

# index = create_index()

def create_tools(index: VectorStoreIndexWrapper, llm) ->List[BaseTool]:
    
    vectorstore_info = VectorStoreInfo(
        name="test_text_code",
        description="A collection of text documents for testing purposes.",
        vectorstore=index.vectorstore,        
    )
    
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
    return toolkit.get_tools()


def chat(message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper) -> str:
    global tools
    if tools is None:
        tools = create_tools(index, llm)
    
    memory = ConversationBufferMemory(
        chat_memory=history,
        memory_key="chat_history",
        return_messages=True
    )
    
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        max_iterations=3,
        verbose=False
    )
    
    try:
        return agent_chain.run(input=message)
    except Exception as e:
        print(f"Error: {e}")
        return "申し訳ありません。もう一度質問してください。"

def respond(message: str, history: list) -> tuple[str, list]:
    chat_history = ChatMessageHistory()
    for human, ai in history:
        chat_history.add_user_message(human)
        chat_history.add_ai_message(ai)
    
    response = chat(message, chat_history, get_index())
    # タプルで2つの値のみを返す
    return response, history + [[message, response]]