"""
build_dataset.py で生成したデータを Qdrant（raiden-main）に登録する

既存の登録スクリプトとの違い:
  - 本文をチャンクに分け、チャンクごとに1件として登録する
    （既存は長文のチャンクのベクトルを平均して1件にしていたため、検索でぼやけやすかった）
  - 図の説明文は本文チャンクにも含め、さらに単独でもトピック名を付けて登録する（type="figure_description"）
  - 同じ doc_id の既存データを削除してから登録するため、再登録しても古いチャンクが残らない
  - CLIP画像ベクトルは登録しない（テキストのベクトルと比較できないため）

使い方（raiden フォルダで実行）:
  venv\\Scripts\\python ingest\\upload_dataset.py ingest\\output\\RAYFace_JP --category rayface --dry-run
  venv\\Scripts\\python ingest\\upload_dataset.py ingest\\output\\RAYFace_JP --category rayface
"""

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient, models
import os

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")

COLLECTION = "raiden-main"
EMBEDDING_MODEL = "text-embedding-3-small"
# 500字が、答えの一文を含むチャンクを上位に拾いやすかった（eval で 1000/700/500/350 を比較）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
WEIGHTS = {"content": 1.0, "figure_description": 0.7}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n【", "\n\n", "\n", "。", "、", " ", ""],
)


def point_id(vector_id: str) -> str:
    # 既存の登録スクリプトと同じ方式（再登録時に同じIDになる）
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, vector_id))


def parse_section_txt(path: Path) -> dict:
    """タグ付きtxt（title[] / topic[] / item[] / 本文 / [FigN]説明）を読み込む"""
    title, topic, items, body_lines, figures = "", "", [], [], {}
    current_fig = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        m = re.fullmatch(r"\[(Fig\d+[A-Za-z0-9]*)\]", s)
        if m:
            current_fig = m.group(1)
            figures[current_fig] = []
        elif current_fig:
            if s:
                figures[current_fig].append(s)
        elif s.startswith("title[") and s.endswith("]"):
            title = s[6:-1].strip()
        elif s.startswith("topic[") and s.endswith("]"):
            topic = s[6:-1].strip()
        elif s.startswith("item[") and s.endswith("]"):
            items.append(s[5:-1].strip())
        else:
            body_lines.append(line)
    return {
        "title": title,
        "topic": topic,
        "items": items,
        "body": "\n".join(body_lines).strip(),
        "figures": {k: "\n".join(v) for k, v in figures.items() if v},
    }


def build_records(doc_dir: Path, category: str) -> tuple[str, list[dict]]:
    records = []
    doc_id = None
    for sec_dir in sorted(p for p in doc_dir.iterdir() if p.is_dir()):
        meta_file = sec_dir / "meta.json"
        txt_file = sec_dir / f"{sec_dir.name}.txt"
        if not meta_file.exists() or not txt_file.exists():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        doc_id = doc_id or meta["doc_id"]
        sec = parse_section_txt(txt_file)
        base = sec_dir.name

        common = {
            "category": category,
            "title": sec["title"],
            "topic": sec["topic"],
            "items": sec["items"],
            "base_name": base,
            "doc_id": meta["doc_id"],
            "source_file": meta["source_file"],
            "source_sha256": meta["source_sha256"],
            "pages": meta["pages"],
            "lang": meta.get("lang", "ja"),
        }
        header = f"title[{sec['title']}]\ntopic[{sec['topic']}]\n\n"
        fig_ids = list(sec["figures"].keys())

        # 図の説明も本文に含めてチャンク化する（本文が検索されれば図の情報も回答に届くように）
        body = sec["body"]
        if sec["figures"]:
            body += "\n\n【図の説明】\n" + "\n".join(f"[{k}] {v}" for k, v in sec["figures"].items())
        chunks = splitter.split_text(body) if body else []
        for i, chunk in enumerate(chunks):
            vid = f"{base}_content" if len(chunks) == 1 else f"{base}_content_{i + 1:02d}"
            records.append({
                "vector_id": vid,
                "payload": {
                    **common,
                    "type": "content",
                    "weight": WEIGHTS["content"],
                    "text": header + chunk,
                    "vector_id": vid,
                    "original_id": vid,
                    "chunk_index": i + 1,
                    "chunk_count": len(chunks),
                    "related_images": [f"{base}_{f}_image" for f in fig_ids],
                    "related_descriptions": [f"{base}_{f}_desc" for f in fig_ids],
                },
            })

        for fig_id, desc in sec["figures"].items():
            vid = f"{base}_{fig_id}_desc"
            records.append({
                "vector_id": vid,
                "payload": {
                    **common,
                    "type": "figure_description",
                    "weight": WEIGHTS["figure_description"],
                    "text": f"【{sec['topic']}の図】\n{desc}",
                    "vector_id": vid,
                    "original_id": vid,
                    "related_content_id": f"{base}_content",
                    "related_image_id": f"{base}_{fig_id}_image",
                },
            })
    return doc_id, records


def embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[i:i + 100])
        vectors += [d.embedding for d in resp.data]
    return vectors


def main():
    parser = argparse.ArgumentParser(description="生成済みデータを Qdrant に登録します")
    parser.add_argument("doc_dir", help="ingest/output/<doc_id> フォルダ")
    parser.add_argument("--category", default="dental", help="payload の category（例: dental, rayface）")
    parser.add_argument("--dry-run", action="store_true", help="登録せず件数と内容だけ表示")
    parser.add_argument("--yes", action="store_true", help="確認なしで登録")
    args = parser.parse_args()

    doc_dir = Path(args.doc_dir)
    doc_id, records = build_records(doc_dir, args.category)
    if not records:
        sys.exit(f"登録するデータがありません: {doc_dir}")

    n_content = sum(r["payload"]["type"] == "content" for r in records)
    n_fig = len(records) - n_content
    lengths = [len(r["payload"]["text"]) for r in records if r["payload"]["type"] == "content"]
    print(f"📦 doc_id={doc_id} / category={args.category} / コレクション={COLLECTION}")
    print(f"  本文チャンク {n_content}件（平均 {sum(lengths) // max(len(lengths), 1)}文字, 最大 {max(lengths, default=0)}文字）"
          f" / 図の説明 {n_fig}件")

    if args.dry_run:
        for r in records[:3]:
            print(f"\n--- {r['vector_id']} ---\n{r['payload']['text'][:300]}")
        return

    qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    # Qdrant Cloud はフィルタに使う項目にインデックスが必要
    indexed = (qdrant.get_collection(COLLECTION).payload_schema or {}).keys()
    for field in ("doc_id", "category", "base_name"):  # base_name は custom.py のセクション展開で使用
        if field not in indexed:
            qdrant.create_payload_index(COLLECTION, field_name=field,
                                        field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
            print(f"  🗂 payload インデックスを作成しました: {field}")
    doc_filter = models.Filter(must=[
        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
    ])
    existing = qdrant.count(COLLECTION, count_filter=doc_filter, exact=True).count
    print(f"  既存の同じ doc_id のデータ: {existing}件（削除してから登録します）")

    if not args.yes:
        if input("\n登録を開始しますか？ (y/n): ").strip().lower() != "y":
            print("中止しました")
            return

    print("  🔢 埋め込みベクトルを生成中...")
    vectors = embed(OpenAI(), [r["payload"]["text"] for r in records])

    if existing:
        qdrant.delete(COLLECTION, points_selector=models.FilterSelector(filter=doc_filter))
        print(f"  🗑 既存の {existing}件を削除しました")

    points = [models.PointStruct(id=point_id(r["vector_id"]), vector=v, payload=r["payload"])
              for r, v in zip(records, vectors)]
    for i in range(0, len(points), 100):
        qdrant.upsert(COLLECTION, points=points[i:i + 100])
    total = qdrant.get_collection(COLLECTION).points_count
    print(f"  ✅ {len(points)}件を登録しました（コレクション総数: {total}）")


if __name__ == "__main__":
    main()
