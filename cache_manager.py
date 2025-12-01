import time
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

# Qdrantクライアントの初期化
def get_qdrant_client():
    """Qdrantクライアントを取得"""
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

def clean_expired_cache(collection_name="raiden-cache", expiration_days=90):
    """有効期限が切れたキャッシュを削除する関数"""
    try:
        print(f"{expiration_days}日以上経過したキャッシュの削除を開始...")
        
        client = get_qdrant_client()
        
        # 現在の日時から有効期限の日時を計算
        cutoff_date = datetime.now() - timedelta(days=expiration_days)
        cutoff_timestamp = cutoff_date.timestamp()
        
        # Qdrantから全ポイントを取得（スクロールAPIを使用）
        offset = None
        expired_ids = []
        
        while True:
            # スクロールでポイントを取得
            result = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_offset = result
            
            # 期限切れのポイントIDを収集
            for point in points:
                if "timestamp" in point.payload:
                    # タイムスタンプを比較（文字列形式の場合）
                    point_timestamp_str = point.payload["timestamp"]
                    point_datetime = datetime.strptime(point_timestamp_str, "%Y-%m-%d %H:%M:%S")
                    
                    if point_datetime < cutoff_date:
                        expired_ids.append(point.id)
            
            # 次のオフセットがない場合は終了
            if next_offset is None:
                break
            offset = next_offset
        
        # 期限切れエントリの削除
        if expired_ids:
            client.delete(
                collection_name=collection_name,
                points_selector=expired_ids
            )
            print(f"{len(expired_ids)}件の期限切れキャッシュを削除しました")
        else:
            print("期限切れのキャッシュはありませんでした")
            
    except Exception as e:
        print(f"キャッシュクリーンアップ中にエラーが発生しました: {e}")

def setup_cache_cleanup_scheduler(collection_name="raiden-cache", expiration_days=90):
    """定期的なキャッシュクリーンアップスケジューラー"""
    scheduler = BackgroundScheduler()
    
    # 毎日午前3時に実行
    scheduler.add_job(
        lambda: clean_expired_cache(collection_name, expiration_days), 
        'cron', 
        hour=3, 
        minute=0
    )
    
    # スケジューラー開始
    scheduler.start()
    print(f"キャッシュクリーンアップスケジューラーを開始しました（有効期限: {expiration_days}日）")
    print(f"対象コレクション: {collection_name}")
    
    # アプリケーション終了時にスケジューラーを停止
    atexit.register(lambda: scheduler.shutdown())

# テスト用の手動実行関数
def manual_cleanup(collection_name="raiden-cache", expiration_days=90):
    """手動でクリーンアップを実行"""
    print("手動クリーンアップを実行します...")
    clean_expired_cache(collection_name, expiration_days)