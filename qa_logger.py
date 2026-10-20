"""
質問と回答を DynamoDB（raiden-qa-logs）に記録する。定期的な答え合わせ（eval/review_logs.py）に使う。

テーブル設計:
  パーティションキー date (YYYY-MM-DD, JST) / ソートキー sk (ISO日時#短いID)
  review_status: unreviewed → auto_reviewed → human_ok / human_ng

記録に失敗してもチャットは止めない（別スレッドで書き込み、例外はログに出すだけ）。
QA_LOG_ENABLED=false で記録を止められる。
"""

import logging
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

logger = logging.getLogger(__name__)

TABLE_NAME = os.getenv("QA_LOG_TABLE", "raiden-qa-logs")
ENABLED = os.getenv("QA_LOG_ENABLED", "true").lower() != "false"
JST = timezone(timedelta(hours=9))
MAX_TEXT = 20000  # DynamoDB の1項目上限（400KB）に余裕を持たせる

_table = None


def _get_table():
    global _table
    if _table is None:
        import boto3
        _table = boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "ap-northeast-1")).Table(TABLE_NAME)
    return _table


def _code_version() -> str:
    """.git を直接読んでコミットIDを得る（本番の systemd は PATH に git がないため git コマンドは使わない）"""
    try:
        git = Path(__file__).parent / ".git"
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head[:7]
        ref = head[5:]
        ref_file = git / ref
        if ref_file.exists():
            return ref_file.read_text().strip()[:7]
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0][:7]
    except Exception:
        pass
    return "unknown"


CODE_VERSION = _code_version()


def _to_dynamo(value):
    """float を Decimal に変換（DynamoDB は float を受け付けないため）"""
    if isinstance(value, float):
        return Decimal(str(round(value, 4)))
    if isinstance(value, dict):
        return {k: _to_dynamo(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_to_dynamo(v) for v in value]
    return value


def _put(item: dict):
    try:
        _get_table().put_item(Item=item)
    except Exception as e:
        logger.warning(f"Q&Aログの保存に失敗しました: {e}")


def log_qa(question: str, answer: str, channel: str, elapsed_sec: float,
           sources: list | None = None, history_turns: int = 0,
           referer: str | None = None, error: str | None = None):
    """質問と回答を記録する（非同期）"""
    if not ENABLED:
        return
    now = datetime.now(JST)
    item = {
        "date": now.strftime("%Y-%m-%d"),
        "sk": f"{now.isoformat(timespec='seconds')}#{uuid.uuid4().hex[:8]}",
        "created_at": now.isoformat(timespec="seconds"),
        "question": (question or "")[:MAX_TEXT],
        "answer": (answer or "")[:MAX_TEXT],
        "channel": channel,
        "elapsed_sec": elapsed_sec,
        "sources": sources or [],
        "history_turns": history_turns,
        "code_version": CODE_VERSION,
        "review_status": "unreviewed",
        "referer": referer,
        "error": error,
    }
    threading.Thread(target=_put, args=(_to_dynamo(item),), daemon=True).start()
