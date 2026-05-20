import time
import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

# ロガーの設定
logger = logging.getLogger(__name__)

# 定数
BATCH_SIZE = 1000
DEFAULT_EXPIRATION_DAYS = 90
DEFAULT_CLEANUP_HOUR = 3
DEFAULT_CLEANUP_MINUTE = 0

def get_qdrant_client():
    """Qdrantクライアントを取得"""
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

def clean_expired_cache(
    collection_name="raiden-cache", 
    expiration_days=DEFAULT_EXPIRATION_DAYS,
    dry_run=False
):
    """有効期限が切れたキャッシュを削除する関数"""
    try:
        mode = "DRY_RUN" if dry_run else "EXECUTE"
        logger.info(f"キャッシュクリーンアップ開始 [{mode}]: {collection_name} (有効期限: {expiration_days}日)")
        
        client = get_qdrant_client()
        
        # コレクションの存在確認
        try:
            collection_info = client.get_collection(collection_name)
            logger.info(f"コレクション確認: {collection_info.points_count}件のポイント")
        except Exception as e:
            logger.warning(f"コレクション '{collection_name}' が見つかりません: {e}")
            return
        
        # 現在の日時から有効期限の日時を計算
        cutoff_date = datetime.now() - timedelta(days=expiration_days)
        logger.info(f"削除対象: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')} 以前のキャッシュ")
        
        # 期限切れポイントIDを収集
        offset = None
        expired_ids = []
        
        while True:
            result = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_offset = result
            
            for point in points:
                if "timestamp" in point.payload:
                    try:
                        point_timestamp_str = point.payload["timestamp"]
                        point_datetime = datetime.strptime(point_timestamp_str, "%Y-%m-%d %H:%M:%S")
                        
                        if point_datetime < cutoff_date:
                            expired_ids.append(point.id)
                    except ValueError as e:
                        logger.warning(f"タイムスタンプ解析エラー (ID: {point.id}): {e}")
            
            if next_offset is None:
                break
            offset = next_offset
        
        # 削除処理
        if expired_ids:
            if dry_run:
                logger.info(f"[DRY_RUN] {len(expired_ids)}件を削除対象として検出")
                for i, exp_id in enumerate(expired_ids[:5]):
                    logger.info(f"  - {exp_id}")
                if len(expired_ids) > 5:
                    logger.info(f"  ... 他 {len(expired_ids) - 5}件")
            else:
                # バッチ削除
                total_deleted = 0
                for i in range(0, len(expired_ids), BATCH_SIZE):
                    batch = expired_ids[i:i+BATCH_SIZE]
                    try:
                        client.delete(
                            collection_name=collection_name,
                            points_selector=batch
                        )
                        total_deleted += len(batch)
                        if len(expired_ids) > BATCH_SIZE:
                            logger.info(f"削除進捗: {total_deleted}/{len(expired_ids)}")
                    except Exception as e:
                        logger.error(f"バッチ削除エラー: {e}")
                
                logger.info(f"✅ 合計 {total_deleted}件のキャッシュを削除しました")
        else:
            logger.info("期限切れのキャッシュはありませんでした")
            
    except Exception as e:
        logger.error(f"❌ キャッシュクリーンアップエラー: {e}", exc_info=True)

def setup_cache_cleanup_scheduler(
    collection_name="raiden-cache", 
    expiration_days=None,
    hour=None,
    minute=None
):
    """定期的なキャッシュクリーンアップスケジューラー"""
    
    # 環境変数から設定を読み込み
    expiration_days = expiration_days or int(os.getenv('CACHE_EXPIRATION_DAYS', str(DEFAULT_EXPIRATION_DAYS)))
    hour = hour or int(os.getenv('CACHE_CLEANUP_HOUR', str(DEFAULT_CLEANUP_HOUR)))
    minute = minute or int(os.getenv('CACHE_CLEANUP_MINUTE', str(DEFAULT_CLEANUP_MINUTE)))
    
    scheduler = BackgroundScheduler()
    
    scheduler.add_job(
        lambda: clean_expired_cache(collection_name, expiration_days), 
        'cron', 
        hour=hour, 
        minute=minute
    )
    
    scheduler.start()
    logger.info(f"✅ スケジューラー開始: 毎日{hour:02d}:{minute:02d}に実行")
    logger.info(f"📅 有効期限: {expiration_days}日")
    logger.info(f"📦 対象コレクション: {collection_name}")
    
    atexit.register(lambda: scheduler.shutdown())

def manual_cleanup(collection_name="raiden-cache", expiration_days=DEFAULT_EXPIRATION_DAYS, dry_run=False):
    """手動でクリーンアップを実行"""
    logger.info("🔧 手動クリーンアップを実行します...")
    clean_expired_cache(collection_name, expiration_days, dry_run)

def get_cache_stats(collection_name="raiden-cache"):
    """キャッシュの統計情報を取得"""
    try:
        client = get_qdrant_client()
        collection_info = client.get_collection(collection_name)
        
        logger.info(f"📊 キャッシュ統計: {collection_name}")
        logger.info(f"  - 総ポイント数: {collection_info.points_count}")
        
        return {"total_points": collection_info.points_count}
        
    except Exception as e:
        logger.error(f"統計情報取得エラー: {e}")
        return None