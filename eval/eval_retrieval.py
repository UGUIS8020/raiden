"""
RAYFace マニュアルの検索性能を測る

各質問について、正解セクション（expect の base_name）が検索結果の何位に入るかを調べる。
本文（content/pubmed_paper）と図の説明（figure_description）を custom.py と同じ条件で検索する。

使い方（raiden フォルダで実行）:
  venv\\Scripts\\python eval\\eval_retrieval.py [--k 15]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdrant_client.models import Filter, FieldCondition, MatchAny  # noqa: E402
from chatbot_engine import get_index  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def search(vs, query, types, k):
    res = vs.similarity_search_with_score(
        query, k=k, filter=Filter(must=[FieldCondition(key="type", match=MatchAny(any=types))])
    )
    # 検索結果には payload が含まれないため、ID から取得して metadata に入れる
    ids = [d.metadata.get("_id") for d, _ in res if d.metadata.get("_id")]
    payloads = {str(p.id): p.payload for p in vs.client.retrieve(vs.collection_name, ids=ids, with_payload=True)}
    for d, _ in res:
        d.metadata.update({k_: v for k_, v in payloads.get(str(d.metadata.get("_id")), {}).items() if k_ != "text"})
    return res


def label(doc):
    return doc.metadata.get("vector_id") or doc.metadata.get("original_id") or "?"


def rank_of(results, expect):
    for i, (doc, _) in enumerate(results, start=1):
        if any(label(doc).startswith(e) for e in expect):
            return i
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--cases", default=str(Path(__file__).with_name("rayface_cases.json")))
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    vs = get_index().vectorstore

    hit1 = hit3 = hitk = n = 0
    for c in cases:
        res = search(vs, c["q"], ["content", "pubmed_paper"], args.k)
        top = res[0][1] if res else 0
        if not c["expect"]:
            print(f"{c['id']:8} (答えなし)  top score={top:.3f}  top={label(res[0][0]) if res else '-'}")
            continue
        n += 1
        r = rank_of(res, c["expect"])
        hit1 += r == 1
        hit3 += r is not None and r <= 3
        hitk += r is not None
        mark = "✅" if r == 1 else ("🟡" if r and r <= 3 else ("🟠" if r else "❌"))
        print(f"{c['id']:8} {mark} rank={r if r else '-':>2}  top score={top:.3f}  top={label(res[0][0])}")

    print(f"\n正解セクションの順位: 1位 {hit1}/{n}  3位以内 {hit3}/{n}  {args.k}位以内 {hitk}/{n}")


if __name__ == "__main__":
    main()
