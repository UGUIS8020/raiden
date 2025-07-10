import time
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# インデックス参照をグローバルにしないための関数
def clean_expired_cache(index, expiration_days=90):
    """有効期限が切れたキャッシュを削除する関数"""
    try:
        print(f"{expiration_days}日以上経過したキャッシュの削除を開始...")
        
        # 現在の日時から有効期限の日時を計算
        cutoff_date = datetime.now() - timedelta(days=expiration_days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Pineconeの全エントリを取得
        results = index.query(
            vector=[0] * 1536,  # ダミーベクトル（実際の次元数に合わせる）
            top_k=10000,        # 十分大きな数
            include_metadata=True
        )
        
        # 期限切れのエントリIDを収集
        expired_ids = []
        for match in results["matches"]:
            metadata = match["metadata"]
            if "timestamp" in metadata:
                if metadata["timestamp"] < cutoff_str:
                    expired_ids.append(match["id"])
        
        # 期限切れエントリの削除
        if expired_ids:
            index.delete(ids=expired_ids)
            print(f"{len(expired_ids)}件の期限切れキャッシュを削除しました")
        else:
            print("期限切れのキャッシュはありませんでした")
            
    except Exception as e:
        print(f"キャッシュクリーンアップ中にエラーが発生しました: {e}")

def setup_cache_cleanup_scheduler(index, expiration_days=90):
    """定期的なキャッシュクリーンアップスケジューラー"""
    scheduler = BackgroundScheduler()
    
    # 毎日午前3時に実行
    scheduler.add_job(
        lambda: clean_expired_cache(index, expiration_days), 
        'cron', 
        hour=3, 
        minute=0
    )
    
    # スケジューラー開始
    scheduler.start()
    print(f"キャッシュクリーンアップスケジューラーを開始しました（有効期限: {expiration_days}日）")
    
    # アプリケーション終了時にスケジューラーを停止
    atexit.register(lambda: scheduler.shutdown())