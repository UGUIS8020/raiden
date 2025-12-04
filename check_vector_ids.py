from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from collections import Counter

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'),
)

def check_vector_id_patterns():
    """vector_idのパターンを確認"""
    print("=== vector_id のパターンを確認 ===\n")
    
    # 最初の50件を取得
    results = client.scroll(
        collection_name="raiden-main",
        limit=50,
        with_payload=['vector_id', 'type', 'title'],
        with_vectors=False
    )
    
    points = results[0]
    
    # プレフィックス（ファイル名部分）を抽出
    prefixes = []
    
    print("サンプルデータ:")
    for i, point in enumerate(points[:20], 1):
        vector_id = point.payload.get('vector_id', '')
        doc_type = point.payload.get('type', 'N/A')
        
        # アンダースコアまたはchapterの前までを抽出
        if '_chapter' in vector_id:
            prefix = vector_id.split('_chapter')[0]
        elif '_ch' in vector_id:
            prefix = vector_id.split('_ch')[0]
        else:
            # 最初のアンダースコアまで
            parts = vector_id.split('_')
            prefix = parts[0] if parts else vector_id
        
        prefixes.append(prefix)
        
        print(f"{i}. vector_id: {vector_id}")
        print(f"   prefix: {prefix}")
        print(f"   type: {doc_type}")
        print()
    
    # プレフィックスの統計
    prefix_counts = Counter(prefixes)
    
    print("\n=== ファイル名プレフィックスの統計 ===")
    for prefix, count in prefix_counts.most_common(10):
        print(f"{prefix}: {count}件")
    
    return prefix_counts

if __name__ == "__main__":
    check_vector_id_patterns()