from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'),
)

# サンプルデータを確認
results = client.scroll(
    collection_name="raiden-main",
    limit=3,
    with_payload=True,
    with_vectors=False
)

print("=== サンプルデータのフィールド ===")
for point in results[0]:
    print(f"\nID: {point.id}")
    print(f"Payload keys: {list(point.payload.keys())}")
    print(f"vector_id: {point.payload.get('vector_id')}")
    print(f"original_id: {point.payload.get('original_id')}")
    break