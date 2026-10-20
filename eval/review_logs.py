"""
Q&Aログの答え合わせ（定期的に実行する）

  1. DynamoDB raiden-qa-logs から未確認（unreviewed）の記録を取り出す
  2. 回答の根拠にした資料（sources）を Qdrant から取得する
     資料の記録がない古いログは、今のデータで検索し直した資料で判定する
  3. 採点役の GPT-4o が「資料に基づいているか」「矛盾はないか」「重要な抜けはないか」を判定する
  4. 判定を DynamoDB に書き戻し（review_status=auto_reviewed）、確認用レポートを出力する
     あわせて、テスト問題集（eval/rayface_cases.json 形式）に追加する候補を出力する

使い方（raiden フォルダで実行）:
  venv\\Scripts\\python eval\\review_logs.py                       # 直近7日の未確認分
  venv\\Scripts\\python eval\\review_logs.py --since 2026-07-01    # 期間を指定
  venv\\Scripts\\python eval\\review_logs.py --dry-run             # DynamoDB に書き戻さない
  # 人が確認した結果を記録する
  venv\\Scripts\\python eval\\review_logs.py --mark 2026-10-21 "2026-10-21T10:00:00+09:00#ab12cd34" ng --note "方向再設定が抜け"

レポートと候補は患者の相談内容を含むため、Git 管理外のフォルダに出力する:
  C:\\Users\\Owner\\Desktop\\website\\_raiden_qa_logs\\reports\\
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import logging  # noqa: E402

import boto3  # noqa: E402
from boto3.dynamodb.conditions import Attr, Key  # noqa: E402
from openai import OpenAI  # noqa: E402

from chatbot_engine import get_index, llm  # noqa: E402
from custom import retrieve_documents, section_text, source_label  # noqa: E402
from qa_logger import JST, TABLE_NAME  # noqa: E402

logging.disable(logging.CRITICAL)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRIVATE_DIR = ROOT.parent / "_raiden_qa_logs"
REPORT_DIR = PRIVATE_DIR / "reports"
MAX_REF_CHARS = 30000

JUDGE_SYSTEM = """あなたは歯科医療・歯科機器のQ&Aシステムの品質管理担当です。
質問、システムの回答、回答の根拠となる資料を読み、回答の品質を厳格に判定してください。
資料に書かれていることだけを正しいとみなし、あなた自身の知識で補わないでください。

資料は2種類ある:「回答時の資料」（システムが回答に使った資料）と「再検索した資料」（同じ質問で改めて検索した資料）。
回答時の資料になく、再検索した資料にある答えを回答が示していなければ、検索の取りこぼしとして missing に含める。

特に次の誤りを見逃さないこと:
- 資料が一般的な機能（例: 角度の計測機能）しか説明していないのに、質問の具体的な用途（例: 咬合平面の角度）に「できる」と断定している → unsupported
- 質問者が示した前提や候補（例: 「AはBでいいですか？」のB）に、資料の裏付けなく同意している → unsupported
- 資料で別々の機能・選択肢として書かれているものを、1つの手順につなげている → contradictions
- 資料にある機能の説明を取り違えている（例: 長さを測る方法を角度の測り方として説明）→ contradictions
- 資料に答えがあるのに「記載なし」と答えている → no_answer_correct=false

手順:
1. 回答から具体的な主張（結論の文、手順の各ステップ、数値、できる/できない、機能の説明）を1つずつ取り出す。
2. 各主張について、それを直接裏付ける資料の文をそのまま引用する（quote）。
   - 主張の言い方が資料より強い・具体的な場合（例: 資料「角度を計測できる」→ 主張「咬合平面の角度を測定できる」）は、裏付けにならない。
   - 引用できなければ status="unsupported"、資料と食い違えば status="contradicted"。
3. その結果から判定項目を決める。

判定項目:
- claims: [{"claim": "回答の主張", "quote": "裏付ける資料の文（なければ null）", "status": "supported|unsupported|contradicted"}]
- unsupported: status="unsupported" の主張（出典表記や一般的なつなぎの文は含めない）。
- contradictions: status="contradicted" の主張。
- missing: 質問に答えるうえで重要なのに回答から抜けている、資料にある情報（画面名・ボタン名・手順・条件・別の方法・別の資料の記載など）。
- answered: 回答が質問に正面から答えているか。
- no_answer_correct: 回答が「記載されていません」等と答えている場合のみ判定。資料に答えがなければ true、あるのに記載なしと答えていれば false。それ以外は null。
- verdict: contradictions がある、または重大な unsupported がある → "wrong"。重要な missing、軽微な unsupported、no_answer_correct=false → "needs_improvement"。それ以外 → "ok"。
- test_case_points: この質問をテスト問題にする場合の「正解のポイント」（資料に基づく、2〜4個の短い箇条）。資料に答えがない質問なら ["資料に記載がないと答える", "一般論や推測で断定しない"]。

出力JSON:
{"claims": [], "verdict": "ok|needs_improvement|wrong", "unsupported": [], "contradictions": [], "missing": [],
 "answered": true, "no_answer_correct": null, "comment": "一言", "test_case_points": []}"""


def daterange(since: date, until: date):
    d = since
    while d <= until:
        yield d.isoformat()
        d += timedelta(days=1)


def fetch_logs(table, since: date, until: date, status: str | None):
    items = []
    for day in daterange(since, until):
        kwargs = {"KeyConditionExpression": Key("date").eq(day)}
        if status:
            kwargs["FilterExpression"] = Attr("review_status").eq(status)
        while True:
            resp = table.query(**kwargs)
            items += resp["Items"]
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


def reference_from_sources(vs, sources):
    """記録された資料（Qdrant の point_id）から、回答時に渡した資料を復元する"""
    ids = [s["point_id"] for s in sources if s.get("point_id")]
    if not ids:
        return []
    points = vs.client.retrieve(vs.collection_name, ids=ids, with_payload=True)
    texts, seen = [], set()
    for p in points:
        pl = p.payload or {}
        base = pl.get("base_name")
        key = base if pl.get("type") == "content" and base else str(p.id)
        if key in seen:
            continue
        seen.add(key)
        body = None
        if pl.get("type") == "content" and base and (pl.get("chunk_count") or 1) > 1:
            body = section_text(vs.client, vs.collection_name, base)
        texts.append(f"{source_label(pl)}\n{body or pl.get('text', '')}")
    return texts


def reference_by_search(vs, question):
    """今のシステムと同じ方法（retrieve_documents）で検索し直す"""
    docs = retrieve_documents(vs, llm, question)
    return [f"{source_label(d.metadata)}\n{d.page_content}" for d in docs]


def judge(client, question, answer, used_refs, searched_refs):
    used = "\n\n---\n\n".join(used_refs)[:MAX_REF_CHARS // 2] or "（記録なし）"
    searched = "\n\n---\n\n".join(searched_refs)[:MAX_REF_CHARS // 2] or "（なし）"
    resp = client.chat.completions.create(
        model="gpt-4o", temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": f"質問:\n{question}\n\nシステムの回答:\n{answer}\n\n"
                                              f"=== 回答時の資料 ===\n{used}\n\n=== 再検索した資料 ===\n{searched}"}],
    )
    return json.loads(resp.choices[0].message.content)


def review_item(vs, client, item):
    sources = item.get("sources") or []
    used_refs = reference_from_sources(vs, sources) if sources else []
    # 検索の取りこぼしも見つけられるよう、今のデータで検索し直した資料も渡す
    searched_refs = [r for r in reference_by_search(vs, item["question"]) if r not in used_refs]
    refs = used_refs + searched_refs
    ref_mode = "回答時の資料＋再検索した資料" if sources else "再検索した資料のみ（回答時の資料の記録なし）"
    if item.get("error") or not item.get("answer"):
        result = {"verdict": "wrong", "unsupported": [], "contradictions": [], "missing": [],
                  "answered": False, "no_answer_correct": None,
                  "comment": f"エラーで回答できていない: {item.get('error', '')[:200]}", "test_case_points": []}
    else:
        result = judge(client, item["question"], item["answer"], used_refs, searched_refs)
    result["reference_mode"] = ref_mode
    result["expect"] = sorted({s.get("base_name") for s in sources if s.get("base_name")})
    return item, result


def write_report(results, path: Path, since, until):
    order = {"wrong": 0, "needs_improvement": 1, "ok": 2}
    results = sorted(results, key=lambda x: (order.get(x[1].get("verdict"), 3), x[0]["sk"]))
    counts = {v: sum(1 for _, r in results if r.get("verdict") == v) for v in order}
    lines = [
        f"# RAIDEN 答え合わせレポート（{since} 〜 {until}）",
        "",
        f"- 対象: {len(results)}件　❌ 誤り {counts['wrong']} / 🟡 要改善 {counts['needs_improvement']} / ✅ 問題なし {counts['ok']}",
        f"- 作成: {datetime.now(JST):%Y-%m-%d %H:%M}",
        "- 人の確認結果を記録する: `venv\\Scripts\\python eval\\review_logs.py --mark <date> \"<sk>\" ok|ng --note \"メモ\"`",
        "",
    ]
    mark = {"wrong": "❌", "needs_improvement": "🟡", "ok": "✅"}
    for item, r in results:
        lines += [
            f"## {mark.get(r.get('verdict'), '?')} {item['question'][:60]}",
            "",
            f"- 日時: {item.get('created_at')}　経路: {item.get('channel')}　応答時間: {item.get('elapsed_sec', '-')}秒"
            f"　判定の根拠: {r.get('reference_mode')}",
            f"- date: `{item['date']}`　sk: `{item['sk']}`",
            f"- 判定: {r.get('comment', '')}",
        ]
        for label, key in [("資料にない記述", "unsupported"), ("資料との矛盾", "contradictions"), ("抜けている情報", "missing")]:
            if r.get(key):
                lines.append(f"- {label}:")
                lines += [f"  - {x}" for x in r[key]]
        if r.get("no_answer_correct") is False:
            lines.append("- ⚠ 資料に答えがあるのに「記載なし」と回答")
        answer = (item.get("answer") or "").replace("\n", "\n  > ")
        lines += ["", "<details><summary>質問と回答の全文</summary>", "",
                  f"**質問**: {item['question']}", "", f"  > {answer}", "", "</details>", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def mark(table, day, sk, status, note):
    table.update_item(
        Key={"date": day, "sk": sk},
        UpdateExpression="SET review_status = :s, human_note = :n, human_reviewed_at = :t",
        ExpressionAttributeValues={":s": f"human_{status}", ":n": note or "",
                                   ":t": datetime.now(JST).isoformat(timespec="seconds")},
        ConditionExpression="attribute_exists(sk)",
    )
    print(f"記録しました: {day} {sk} → human_{status}")


def main():
    parser = argparse.ArgumentParser(description="RAIDEN の Q&A ログを答え合わせする")
    parser.add_argument("--since", help="開始日 YYYY-MM-DD（既定: 7日前）")
    parser.add_argument("--until", help="終了日 YYYY-MM-DD（既定: 今日）")
    parser.add_argument("--all", action="store_true", help="確認済みも含めて再判定する")
    parser.add_argument("--dry-run", action="store_true", help="DynamoDB に書き戻さない")
    parser.add_argument("--mark", nargs=3, metavar=("DATE", "SK", "ok|ng"), help="人の確認結果を記録する")
    parser.add_argument("--note", help="--mark と一緒に使うメモ")
    args = parser.parse_args()

    table = boto3.resource("dynamodb", region_name="ap-northeast-1").Table(TABLE_NAME)
    if args.mark:
        day, sk, status = args.mark
        if status not in ("ok", "ng"):
            parser.error("ok か ng を指定してください")
        mark(table, day, sk, status, args.note)
        return

    today = datetime.now(JST).date()
    since = date.fromisoformat(args.since) if args.since else today - timedelta(days=7)
    until = date.fromisoformat(args.until) if args.until else today
    items = fetch_logs(table, since, until, None if args.all else "unreviewed")
    print(f"対象: {since} 〜 {until} の{'全' if args.all else '未確認の'}記録 {len(items)}件")
    if not items:
        return

    vs = get_index().vectorstore
    client = OpenAI()
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda it: review_item(vs, client, it), items))

    now = datetime.now(JST)
    if not args.dry_run:
        for item, r in results:
            table.update_item(
                Key={"date": item["date"], "sk": item["sk"]},
                UpdateExpression="SET review_status = :s, auto_review = :r, auto_reviewed_at = :t",
                ExpressionAttributeValues={":s": "auto_reviewed", ":r": r, ":t": now.isoformat(timespec="seconds")},
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / f"review_{now:%Y%m%d_%H%M}.md"
    write_report(results, report, since, until)

    candidates = [{"id": f"log-{it['sk'][:19]}", "q": it["question"], "expect": r["expect"],
                   "points": r.get("test_case_points") or [], "verdict": r.get("verdict")}
                  for it, r in results if r.get("test_case_points")]
    cand_path = REPORT_DIR / f"candidate_cases_{now:%Y%m%d_%H%M}.json"
    cand_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    for v, m in [("wrong", "❌ 誤り"), ("needs_improvement", "🟡 要改善"), ("ok", "✅ 問題なし")]:
        print(f"  {m}: {sum(1 for _, r in results if r.get('verdict') == v)}件")
    print(f"レポート: {report}")
    print(f"テスト問題の候補: {cand_path}")
    if args.dry_run:
        print("（--dry-run のため DynamoDB には書き戻していません）")


if __name__ == "__main__":
    main()
