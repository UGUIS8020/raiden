import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as PineconeVectorStore
import time
import re
import hashlib
from typing import Dict, Optional, Any, List
import json
import openai
import uuid

# 環境変数の読み込み
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# 定数
CACHE_INDEX_NAME = "raiden-cache"  # 既存のキャッシュインデックス名
EMBEDDING_MODEL = "text-embedding-3-small"  # 埋め込みモデル
SIMILARITY_THRESHOLD = 0.85  # 類似度閾値

  # インデックスの統計情報を表示


# デバッグモード
DEBUG_MODE = True

def log_debug(message):
    """デバッグログを出力する"""
    if DEBUG_MODE:
        print(f"[CACHE] {message}")

# Pineconeクライアントの初期化
pc = Pinecone(api_key=PINECONE_API_KEY)

# OpenAI Embeddingsの初期化
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# キャッシュインデックス変数
_cache_vectorstore = None

def enrich_metadata_with_ai(question: str, answer: str) -> Dict[str, Any]:
    """質問と回答からAIを使ってメタデータを補完する"""
    prompt = f"""
    以下の質問と回答から、次の情報をJSON形式で出力してください。
    - category: 質問のカテゴリ（例: 歯科治療, インプラント, 歯科技工所）
    - synonyms: 質問の表記ゆれや同義語のリスト（例: ["こしがや", "越谷市"]）
    - keywords: 質問と回答に関連するキーワード（例: ["移植", "インプラント", "歯根"]）
    - tags: 質問のタグ（例: ["越谷", "歯科", "技工所情報"]）

    質問: {question}
    回答: {answer}

    出力は必ずJSON形式のみでお願いします。
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo" ifコストが気になるなら
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        content = response.choices[0].message['content']
        ai_metadata = json.loads(content)
        
        log_debug("AIメタデータ生成結果:")
        log_debug(ai_metadata)
        
        return ai_metadata
        
    except Exception as e:
        log_debug(f"AIメタデータ補完エラー: {e}")
        # AIエラー時は空のデフォルトを返す
        return {
            "category": "未分類",
            "synonyms": [],
            "keywords": [],
            "tags": []
        }
    
def get_cache_vectorstore():
    """キャッシュ用のベクトルストアを取得する"""
    global _cache_vectorstore
    
    if _cache_vectorstore is not None:
        return _cache_vectorstore
        
    try:
        # インデックスの存在確認
        indices = pc.list_indexes()
        index_names = [index.name for index in indices]
        
        if CACHE_INDEX_NAME not in index_names:
            log_debug(f"エラー: キャッシュインデックス '{CACHE_INDEX_NAME}' が存在しません")
            return None
            
        # ベクトルストアをインスタンス化
        _cache_vectorstore = PineconeVectorStore.from_existing_index(
            index_name=CACHE_INDEX_NAME,
            embedding=embeddings,
            text_key="question"
        )
        
        # インデックスの統計情報を表示
        cache_index = pc.Index(CACHE_INDEX_NAME)
        stats = cache_index.describe_index_stats()
        log_debug(f"キャッシュインデックス '{CACHE_INDEX_NAME}' の統計:")
        log_debug(f"- ベクター数: {stats.total_vector_count}")
        log_debug(f"- 次元数: {stats.dimension}")
        
        return _cache_vectorstore
        
    except Exception as e:
        log_debug(f"キャッシュベクトルストアの取得中にエラー: {e}")
        return None

def normalize_question(question: str) -> str:
    """質問テキストを正規化する"""
    # 句読点や特殊文字を削除
    normalized = re.sub(r'[.,?!;:"\'。、？！；：""'']', '', question)
    # 「は何ですか」「を教えてください」などの表現を削除
    normalized = re.sub(r'(は何(です|でしょう)か[？\?]?|を教えて(ください|下さい)|について(教えて|説明して)(ください|下さい))', '', normalized)
    # 空白を削除
    normalized = normalized.replace(' ', '')
    return normalized.strip()


question = "自家歯牙移植のメリットとデメリットを教えてください"
answer = "メリットは○○、デメリットは××"

normalized_question = normalize_question(question)
question_id = str(uuid.uuid4())

ai_metadata = enrich_metadata_with_ai(question, answer)

metadata = {
    "original_question": question,
    "normalized_question": normalized_question,
    "answer": answer,
    "timestamp": time.time(),
    "category": ai_metadata.get("category", "未分類"),
    "synonyms": ai_metadata.get("synonyms", []),
    "keywords": ai_metadata.get("keywords", []),
    "tags": ai_metadata.get("tags", [])
}

vectorstore = get_cache_vectorstore()

if vectorstore:
    result = vectorstore.add_texts(
        texts=[normalized_question],
        metadatas=[metadata],
        ids=[question_id]
    )
    print(f"保存結果: {result}")
else:
    print("ベクトルストアが取得できませんでした")