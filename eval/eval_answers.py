"""
RAYFace マニュアルの質問に対する RAIDEN の回答を採点する

各質問を chat() に投げ、採点役の GPT-4o が次の観点で判定する:
  - 正解のポイント（rayface_cases.json の points）をいくつ押さえているか
  - マニュアル（正解セクションの原文）にないことを書いていないか
  - 答えがない質問（expect が空）で「記載なし」と答えられたか

使い方（raiden フォルダで実行）:
  venv\\Scripts\\python eval\\eval_answers.py --label before
結果は eval/results/<label>.json に保存される。
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging  # noqa: E402
from langchain_community.chat_message_histories import ChatMessageHistory  # noqa: E402
from openai import OpenAI  # noqa: E402
from chatbot_engine import chat, get_index  # noqa: E402
from prompts import SYSTEM_PROMPT_TEMPLATE  # noqa: E402

logging.disable(logging.CRITICAL)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT = ROOT / "ingest" / "output"
JUDGE_SYSTEM = """あなたは歯科機器マニュアルQAシステムの採点者です。厳格に判定してください。
入力: 質問、正解のポイント、マニュアルの該当箇所（原文）、システムの回答。

判定:
1. points: 各正解ポイントについて、回答がその内容を実質的に含んでいれば true。画面名・ボタン名が求められているポイントは、その名前が回答に出ていなければ false。
2. unsupported: 回答の中で、マニュアルの該当箇所から裏付けられない具体的な主張（手順・数値・機能・効果など）を列挙する。一般的なつなぎの文や「サポートに問い合わせてください」程度の案内は含めない。
3. no_answer_ok: 正解のポイントが「記載がないと答える」系の場合のみ判定。回答が、文書に記載がないことを明示し、かつ具体的な内容をでっち上げていなければ true。それ以外の質問では null。

出力JSON: {"points": [true, false, ...], "unsupported": ["..."], "no_answer_ok": true|false|null, "comment": "一言"}"""


def _section_meta():
    metas = []
    for meta_file in OUTPUT.glob("*/*/meta.json"):
        m = json.loads(meta_file.read_text(encoding="utf-8"))
        metas.append((m["source_file"], m["pages"][0], m["pages"][1], meta_file.parent.name))
    return metas


SECTIONS = _section_meta()


def cited_sections(answer):
    """回答中の【出典: ファイル名 p.a-b】から、該当するセクションを探す"""
    found = []
    for m in re.finditer(r"【出典: (.+?) p\.(\d+)(?:-(\d+))?", answer):
        name, a = m.group(1).strip(), int(m.group(2))
        b = int(m.group(3) or a)
        found += [base for src, s, e, base in SECTIONS if src == name and s <= b and a <= e]
    return found


def reference_text(expect, answer=""):
    texts = []
    for base in dict.fromkeys(list(expect) + cited_sections(answer)):  # 想定の正解 + 回答が挙げた出典
        doc = base.rsplit("_", 1)[0]
        f = OUTPUT / doc / base / f"{base}.txt"
        if f.exists():
            texts.append(f.read_text(encoding="utf-8"))
    return "\n\n".join(texts) or "（該当箇所なし: マニュアルに記載がない質問）"


def judge(client, case, answer):
    content = (
        f"質問: {case['q']}\n\n正解のポイント:\n"
        + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(case["points"]))
        + f"\n\nマニュアルの該当箇所:\n{reference_text(case['expect'], answer)}\n\nシステムの回答:\n{answer}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o", temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": content}],
    )
    return json.loads(resp.choices[0].message.content)


def run_case(index, client, case):
    prompt = SYSTEM_PROMPT_TEMPLATE.format(question=case["q"])
    answer = chat(prompt, ChatMessageHistory(), index)
    j = judge(client, case, answer)
    pts = j.get("points") or []
    return {**case, "answer": answer, "judge": j,
            "hit": sum(bool(p) for p in pts), "total": len(case["points"]),
            "unsupported_n": len(j.get("unsupported") or [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--cases", default=str(Path(__file__).with_name("rayface_cases.json")))
    parser.add_argument("--only", help="カンマ区切りの case id")
    parser.add_argument("--rejudge", help="保存済みの結果（label）の回答を、今の採点方法で採点し直す")
    args = parser.parse_args()

    out_dir = Path(__file__).with_name("results")
    client = OpenAI()
    if args.rejudge:
        saved = json.loads((out_dir / f"{args.rejudge}.json").read_text(encoding="utf-8"))
        cases_by_id = {c["id"]: c for c in json.loads(Path(args.cases).read_text(encoding="utf-8"))}

        def rejudge(r):
            case = cases_by_id[r["id"]]
            j = judge(client, case, r["answer"])
            return {**case, "answer": r["answer"], "judge": j, "hit": sum(bool(p) for p in (j.get("points") or [])),
                    "total": len(case["points"]), "unsupported_n": len(j.get("unsupported") or [])}

        with ThreadPoolExecutor(max_workers=6) as ex:
            results = list(ex.map(rejudge, saved))
    else:
        cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        if args.only:
            cases = [c for c in cases if c["id"] in args.only.split(",")]
        index = get_index()
        run_case(index, client, cases[0])  # ツールの初期化を先に1回だけ行う（並列実行時の競合を避ける）
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(lambda c: run_case(index, client, c), cases))

    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{args.label}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    hit = sum(r["hit"] for r in results if r["expect"])
    total = sum(r["total"] for r in results if r["expect"])
    unsup = sum(r["unsupported_n"] for r in results)
    full = sum(1 for r in results if r["expect"] and r["hit"] == r["total"])
    neg = [r for r in results if not r["expect"]]
    neg_ok = sum(1 for r in neg if r["judge"].get("no_answer_ok"))
    for r in results:
        mark = ("✅" if r["hit"] == r["total"] else "🟡" if r["hit"] else "❌") if r["expect"] else \
               ("✅" if r["judge"].get("no_answer_ok") else "❌")
        print(f"{r['id']:8} {mark} {r['hit']}/{r['total']}  根拠なし:{r['unsupported_n']}  {r['judge'].get('comment', '')[:60]}")
    print(f"\n[{args.label}] ポイント {hit}/{total} ({hit / total:.0%})  全ポイント正解 {full}/{len(results) - len(neg)}"
          f"  根拠のない記述 {unsup}件  記載なしと答えられた {neg_ok}/{len(neg)}")


if __name__ == "__main__":
    main()
