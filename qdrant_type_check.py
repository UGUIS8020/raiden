# 確認用スクリプト
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'),
)

# ランダムに1件取得してpayloadを確認
result = client.scroll(
    collection_name="raiden-main",
    limit=1,
    with_payload=True,
    with_vectors=False
)

print("サンプルデータのpayload:")
if result[0]:
    print(result[0][0].payload)