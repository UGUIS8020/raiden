import langchain
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain.agents.agent_toolkits import VectorStoreToolkit, VectorStoreInfo

from typing import List
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.text_splitter import CharacterTextSplitter

langchain.verbose = True

load_dotenv()

# langsmithを使うためのコード
openai_api_key = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = "LangSmith-test"



def create_index() -> VectorStoreIndexWrapper:
    # テキスト分割機能の設定
    splitter = CharacterTextSplitter(separator="。", chunk_size=100, chunk_overlap=0)

    # DirectoryLoader の初期化
    loader = DirectoryLoader("text/", glob="**/*.txt")
    documents = loader.load()

    # テキストを分割
    split_docs = splitter.split_documents(documents)

    # OpenAI の埋め込みモデル
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-large")

    # Chroma VectorStore の初期化とデータの追加
    vectorstore = Chroma(embedding_function=embedding, persist_directory="chroma_storage")
    vectorstore.add_documents(split_docs)

    # VectorStoreIndexWrapper の作成
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)

    return index

index = create_index()

def create_tools(index: VectorStoreIndexWrapper, llm) ->List[BaseTool]:
    
    vectorstore_info = VectorStoreInfo(
        name="test_text_code",
        description="A collection of text documents for testing purposes.",
        vectorstore=index.vectorstore,        
    )
    
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
    return toolkit.get_tools()


def chat(message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper) -> str:
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    tools = create_tools(index, llm)
    memory = ConversationBufferMemory(chat_memory=history, memory_key="chat_history", return_messages=True)
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory)
    
   
    return agent_chain.run(input=message)