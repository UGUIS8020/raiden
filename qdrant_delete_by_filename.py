from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'),
)

def search_by_filename(filename_pattern):
    """ファイル名パターンで検索"""
    print(f"=== 検索: '{filename_pattern}' を含むデータ ===\n")
    
    # 全データを取得
    all_points = []
    offset = None
    
    while True:
        results = client.scroll(
            collection_name="raiden-main",
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        points, offset = results
        all_points.extend(points)
        
        if offset is None:
            break
    
    print(f"総データ数: {len(all_points)}件")
    
    # ファイル名パターンでフィルタ
    matched_points = []
    for point in all_points:
        vector_id = point.payload.get('vector_id', '')
        
        if filename_pattern in vector_id:
            matched_points.append(point)
    
    print(f"マッチした件数: {len(matched_points)}件\n")
    
    # マッチしたデータを表示
    type_count = {}
    for point in matched_points:
        doc_type = point.payload.get('type', 'N/A')
        type_count[doc_type] = type_count.get(doc_type, 0) + 1
    
    print("タイプ別件数:")
    for doc_type, count in type_count.items():
        print(f"  {doc_type}: {count}件")
    
    print("\n最初の10件:")
    for i, point in enumerate(matched_points[:10], 1):
        print(f"{i}. vector_id: {point.payload.get('vector_id')}")
        print(f"   type: {point.payload.get('type')}")
        print(f"   title: {point.payload.get('title', 'N/A')}")
        print()
    
    if len(matched_points) > 10:
        print(f"... 他 {len(matched_points) - 10}件")
    
    return matched_points

def delete_points(points):
    """ポイントを削除"""
    if not points:
        print("削除するデータがありません。")
        return
    
    point_ids = [str(point.id) for point in points]
    
    print(f"\n⚠️  {len(point_ids)}件のデータを削除します。")
    print("\n削除されるデータの例:")
    for i, point in enumerate(points[:5], 1):
        print(f"  {i}. {point.payload.get('vector_id')} ({point.payload.get('type')})")
    
    if len(points) > 5:
        print(f"  ... 他 {len(points) - 5}件")
    
    response = input("\n本当に削除しますか? (yes/no): ")
    if response.lower() != 'yes':
        print("キャンセルしました。")
        return
    
    # 削除実行
    client.delete(
        collection_name="raiden-main",
        points_selector=point_ids
    )
    
    print(f"✅ {len(point_ids)}件のデータを削除しました。")

def main():
    print("=== Qdrant データ削除ツール ===\n")
    
    # ファイル名パターンを入力
    filename = input("削除したいファイル名のパターンを入力 (例: Transplantation_quint201901): ")
    
    if not filename:
        print("ファイル名が入力されていません。")
        return
    
    # 検索
    matched_points = search_by_filename(filename)
    
    if not matched_points:
        print("\n該当するデータが見つかりませんでした。")
        return
    
    # 削除オプション
    print("\n削除オプションを選択:")
    print("1. 全て削除")
    print("2. type='content'のみ削除")
    print("3. type='figure_description'のみ削除")
    print("4. type='image'のみ削除")
    print("5. キャンセル")
    
    choice = input("\n選択 (1-5): ")
    
    if choice == "1":
        delete_points(matched_points)
    elif choice == "2":
        content_points = [p for p in matched_points if p.payload.get('type') == 'content']
        print(f"\ntype='content': {len(content_points)}件")
        delete_points(content_points)
    elif choice == "3":
        figure_points = [p for p in matched_points if p.payload.get('type') == 'figure_description']
        print(f"\ntype='figure_description': {len(figure_points)}件")
        delete_points(figure_points)
    elif choice == "4":
        image_points = [p for p in matched_points if p.payload.get('type') == 'image']
        print(f"\ntype='image': {len(image_points)}件")
        delete_points(image_points)
    else:
        print("キャンセルしました。")

if __name__ == "__main__":
    main()