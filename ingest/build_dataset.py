"""
RAG用データセット自動生成

PDF / txt を読み込み、既存の登録スクリプト
（CLIPembedding/text_image_embedding_qdrant.py）と同じ形式の
「タグ付きtxt + Fig画像」を自動生成する。

処理の流れ:
  1. 抽出    : ページごとのテキスト・図（画像）を取り出す。文字が取れないページはGPT-4oで読み取る
  2. 分割    : GPT-4oが文書全体を見て、章・節の切れ目でセクションに分ける
  3. 構造化  : セクションごとに本文を整形（英語などは日本語に翻訳）し、topic / item を付ける
  4. 図の説明: 図ごとに説明文を作成し、ロゴ等の不要な図は除外する
  5. 出力    : output/<doc_id>/<base_name>/ に txt・Fig画像・meta.json を書き出す

使い方（raiden フォルダで実行）:
  # APIを使わずに抽出結果だけ確認
  venv\\Scripts\\python ingest\\build_dataset.py "..\\rayface\\manual.pdf" --doc-id RAYFace_JP --dry-run

  # 生成（まず数セクションだけ試す場合は --max-sections）
  venv\\Scripts\\python ingest\\build_dataset.py "..\\rayface\\manual.pdf" --doc-id RAYFace_JP --max-sections 3

LLMの応答は ingest/.cache にキャッシュされるため、同じ入力で再実行しても費用はかからない。
"""

import argparse
import base64
import hashlib
import io
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")

MODEL = "gpt-4o"
PROMPT_VERSION = "v3"  # プロンプトを変更したら上げる（キャッシュを無効化するため）
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / ".cache"

MIN_FIG_PT = 60          # 短辺がこれ未満の画像（ボタン・アイコン等）は図として扱わない
MAX_REPEAT_PAGES = 3     # これを超えるページに繰り返し出る画像はロゴ等とみなして除外
MAX_SECTION_PAGES = 4    # 1セクションの最大ページ数
MAX_SECTION_CHARS = 3000 # 1セクションの原文の最大文字数（多すぎるとLLMが内容を省略しやすい）
FIG_BATCH = 8            # 図の説明を1回のAPI呼び出しで処理する枚数
OCR_MIN_CHARS = 30       # 抽出文字数がこれ未満のページは画像から文字を読み取る
TXT_PAGE_CHARS = 2000    # txt入力を疑似ページに分ける文字数
WORKERS = 4

# 料金の目安（USD / 100万トークン）
PRICE_INPUT = 2.50
PRICE_OUTPUT = 10.00

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


# ========== データ構造 ==========

@dataclass
class Figure:
    page: int      # 1始まり
    bbox: tuple    # PDF上の座標(pt)
    jpeg: bytes


@dataclass
class Page:
    no: int                    # 1始まり
    text: str
    render: bytes | None = None   # レイアウト確認用のページ画像
    figures: list = field(default_factory=list)


# ========== 1. 抽出 ==========

def render_jpeg(page, rect, dpi, max_side=1600) -> bytes:
    pix = page.get_pixmap(clip=rect, dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def load_pdf(path: Path) -> list[Page]:
    doc = fitz.open(path)

    # 画像ごとの出現ページ（ロゴなど繰り返し出る画像の判定用）
    xref_pages: dict[int, set] = {}
    for page in doc:
        for info in page.get_image_info(xrefs=True):
            if info["xref"]:
                xref_pages.setdefault(info["xref"], set()).add(page.number)

    seen_xrefs = set()
    pages = []
    for page in doc:
        figures = []
        kept_rects = []
        infos = sorted(page.get_image_info(xrefs=True),
                       key=lambda i: (round(i["bbox"][1]), i["bbox"][0]))
        for info in infos:
            xref = info["xref"]
            rect = fitz.Rect(info["bbox"]) & page.rect
            if rect.is_empty or min(rect.width, rect.height) < MIN_FIG_PT:
                continue
            if xref and len(xref_pages.get(xref, ())) > MAX_REPEAT_PAGES:
                continue
            if xref and xref in seen_xrefs:
                continue
            # 既に採用した図の内側にある画像は、その図の切り出しに含まれるので除外
            if any(r.contains(rect) for r in kept_rects):
                continue
            if xref:
                seen_xrefs.add(xref)
            kept_rects.append(rect)
            figures.append(Figure(
                page=page.number + 1,
                bbox=tuple(round(v, 1) for v in rect),
                jpeg=render_jpeg(page, rect, dpi=200),
            ))

        pages.append(Page(
            no=page.number + 1,
            text=page.get_text("text"),
            render=render_jpeg(page, page.rect, dpi=100),
            figures=figures,
        ))

    strip_repeated_lines(pages)
    return pages


def load_txt(path: Path) -> list[Page]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp932")

    # 段落単位で疑似ページにまとめる
    pages, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        if buf and len(buf) + len(para) > TXT_PAGE_CHARS:
            pages.append(Page(no=len(pages) + 1, text=buf))
            buf = ""
        buf += para.strip() + "\n\n"
    if buf.strip():
        pages.append(Page(no=len(pages) + 1, text=buf))
    return pages


def strip_repeated_lines(pages: list[Page]):
    """多くのページに共通して出る行（ヘッダー・フッター）とページ番号だけの行を除去"""
    if len(pages) < 5:
        return

    def norm(line):
        return re.sub(r"\d+", "#", line.strip())

    counts = Counter()
    for p in pages:
        counts.update({norm(l) for l in p.text.splitlines() if l.strip()})
    common = {k for k, c in counts.items() if c >= len(pages) * 0.4 and len(k) <= 60}

    for p in pages:
        p.text = "\n".join(
            l for l in p.text.splitlines()
            if l.strip() and norm(l) not in common and not re.fullmatch(r"\s*\d+\s*", l)
        )


# ========== LLM呼び出し ==========

_client = None
_usage = Counter()
_usage_lock = threading.Lock()


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(jpeg: bytes, detail: str = "auto") -> dict:
    b64 = base64.b64encode(jpeg).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}}


def llm_json(system: str, content: list, max_tokens: int = 4096) -> dict:
    """JSONで応答を得る。同じ入力の結果はキャッシュから返す"""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    key = hashlib.sha256(
        json.dumps([MODEL, PROMPT_VERSION, messages], ensure_ascii=False).encode()
    ).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with _usage_lock:
            _usage["cache_hits"] += 1
        return json.loads(cache_file.read_text("utf-8"))

    for attempt in range(3):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                raise ValueError("出力が上限に達しました（MAX_SECTION_PAGES を減らしてください）")
            data = json.loads(choice.message.content)
            with _usage_lock:
                _usage["calls"] += 1
                _usage["input_tokens"] += resp.usage.prompt_tokens
                _usage["output_tokens"] += resp.usage.completion_tokens
            break
        except ValueError:
            raise
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  ⚠ API呼び出し失敗（再試行 {attempt + 1}/2）: {e}")
            time.sleep(5 * (attempt + 1))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    return data


# ========== 文字が取れないページの読み取り ==========

OCR_SYSTEM = """あなたは文書のページ画像から文字を正確に書き起こすアシスタントです。
画像内の文字をすべて、読む順序どおりに書き起こしてください。要約や補足はしないでください。
出力JSON: {"text": "書き起こした文字"}"""


def ocr_missing_text(pages: list[Page]):
    for p in pages:
        if p.render and len(p.text.strip()) < OCR_MIN_CHARS:
            print(f"  🔎 p{p.no}: 文字が少ないため画像から読み取ります")
            p.text = llm_json(OCR_SYSTEM, [image_part(p.render)]).get("text", "")


# ========== 2. セクション分割 ==========

OUTLINE_SYSTEM = f"""あなたは文書をRAG（検索拡張生成）用に分割する編集者です。
ページごとの抽出テキストを読み、内容のまとまり（章・節の見出しの切れ目）ごとにセクションに分けてください。

規則:
- 全ページを先頭から順に、漏れなく重複なく割り当てる。
- 1セクションは1〜{MAX_SECTION_PAGES}ページ。それより長い章は節の切れ目で分ける。
- 章（番号付きの大見出し）が変わるページでは必ず区切り、異なる章を1つのセクションに混ぜない。
- 表紙・目次・白紙・著作権表示など、質問への回答に役立たないページは skip=true のセクションにする。
  問い合わせ先・サポート窓口・仕様・注意事項のページは回答に使えるので skip にしない。

出力JSON:
{{"doc_title": "文書全体を表す短い日本語タイトル（製品名・書名など）",
  "lang": "文書の主な言語（ja / en など）",
  "sections": [{{"start_page": 1, "end_page": 2, "topic": "セクションの主題（日本語・30字以内）", "skip": false}}]}}"""


def make_outline(pages: list[Page], file_name: str) -> dict:
    per_page_limit = max(300, 100_000 // max(len(pages), 1))
    body = "\n\n".join(f"=== p{p.no} ===\n{p.text[:per_page_limit]}" for p in pages)
    content = [text_part(f"ファイル名: {file_name}\n総ページ数: {len(pages)}\n\n{body}")]
    outline = llm_json(OUTLINE_SYSTEM, content)
    outline["sections"] = normalize_sections(outline.get("sections", []), pages)
    return outline


def normalize_sections(raw: list, pages: list[Page]) -> list[dict]:
    """LLMの分割結果を、全ページを順に漏れなく覆う形に補正する"""
    n_pages = len(pages)
    valid = []
    for s in raw:
        try:
            valid.append({
                "start_page": int(s["start_page"]),
                "end_page": int(s["end_page"]),
                "topic": str(s.get("topic", "")),
                "skip": bool(s.get("skip", False)),
            })
        except (KeyError, TypeError, ValueError):
            continue
    valid.sort(key=lambda s: s["start_page"])

    result, next_page = [], 1
    for s in valid:
        start, end = max(s["start_page"], next_page), min(s["end_page"], n_pages)
        if start > end:
            continue
        if start > next_page:  # 抜けているページは直前のセクションに含める
            if result:
                result[-1]["end_page"] = start - 1
            else:
                result.append({"start_page": next_page, "end_page": start - 1, "topic": "", "skip": False})
        result.append({**s, "start_page": start, "end_page": end})
        next_page = end + 1
    if next_page <= n_pages:
        if result:
            result[-1]["end_page"] = n_pages
        else:
            result.append({"start_page": 1, "end_page": n_pages, "topic": "", "skip": False})

    # ページ数・文字数が多すぎるセクションは分割
    page_chars = {p.no: len(p.text) for p in pages}
    split = []
    for s in result:
        start = s["start_page"]
        chars = 0
        for no in range(s["start_page"], s["end_page"] + 1):
            over_pages = no - start >= MAX_SECTION_PAGES
            over_chars = no > start and chars + page_chars[no] > MAX_SECTION_CHARS
            if over_pages or over_chars:
                split.append({**s, "start_page": start, "end_page": no - 1})
                start, chars = no, 0
            chars += page_chars[no]
        split.append({**s, "start_page": start, "end_page": s["end_page"]})
    return split


# ========== 3. 本文の構造化 ==========

STRUCTURE_SYSTEM = """あなたは歯科医療分野のRAG（検索拡張生成）用ナレッジベースを作る編集者です。
文書の一部（抽出テキストとページ画像）から、検索と回答生成に使う本文を作成してください。

規則:
- 本文は日本語で書く。原文が日本語以外の場合は正確に日本語へ翻訳し、ボタン名・メニュー名などの画面表記は原文の表記を「」で併記する。
- 原文の情報（数値・手順・条件・注意事項・警告）は省略せずすべて残す。要約しない。
  例外条件・補足（「または〜の場合」「〜を超える場合」など）も必ず残す。
- 表は1行ずつ、すべての行・列の内容を漏れなく文章にする（例: 「メッセージ」 原因: … 対処: …）。
- 画面に表示されるメッセージ・エラー文・ボタン名・メニュー名は、利用者がそのまま検索できるよう原文の表記を「」で必ず残す（翻訳する場合も併記）。
- 原文にない情報を追加しない。ページ画像は、抽出テキストで欠けているアイコン名・ボタン名・配置の補完にだけ使う（画像から明確に読み取れる場合に限る）。
- PDF抽出で生じた不自然な改行・文字の間の空白・ヘッダー・フッター・ページ番号は取り除く。
- 手順は「1. 2. 3.」の番号付き、小見出しは【】で示す。Markdownの # や表記法は使わない。
- 行の先頭を「[」で始めない。

出力JSON:
{"topic": "このセクションの主題（日本語・40字以内）",
 "items": ["検索に役立つ重要語（最大5個）"],
 "body": "本文"}"""


def structure_section(doc_title: str, file_name: str, sec: dict, pages: list[Page]) -> dict:
    text = "\n\n".join(f"=== p{p.no} ===\n{p.text}" for p in pages)
    content = [text_part(
        f"文書: {doc_title}\nファイル: {file_name}\n"
        f"ページ: {sec['start_page']}-{sec['end_page']}\n候補トピック: {sec['topic']}\n\n"
        f"=== 抽出テキスト ===\n{text}"
    )]
    for p in pages:
        if p.render:
            content.append(text_part(f"p{p.no} のページ画像:"))
            content.append(image_part(p.render))
    return llm_json(STRUCTURE_SYSTEM, content, max_tokens=12000)


# ========== 4. 図の説明 ==========

FIGURE_SYSTEM = """あなたは歯科医療分野のRAG用ナレッジベースを作る編集者です。
文書から切り出した図それぞれについて、検索と回答生成に使う説明文を日本語で作成してください。

規則:
- 図に写っている内容を具体的に書く（画面名、ボタン・アイコン、操作の状態。臨床写真なら部位・所見）。
- 図の中に読める文字（画面タイトル、ボタン名、ラベル、数値）は、検索で見つかるようにすべてそのままの表記で書き出す。
- 赤枠・矢印などで強調されている箇所があれば、それが何かを書く。
- 本文の文脈から、その図が何を示すためのものかを書く。
- 図からも本文からも読み取れないことは推測で書かない。
- 2〜6文程度。
- ロゴ・装飾・意味のない小さなアイコン・ほぼ同じ内容の図など、検索に役立たないものは keep=false にする。

出力JSON: {"figures": [{"index": 1, "keep": true, "description": "説明文"}]}"""


def describe_figures(doc_title: str, topic: str, body: str, figures: list[Figure]) -> list[tuple[Figure, str]]:
    kept = []
    for start in range(0, len(figures), FIG_BATCH):
        batch = figures[start:start + FIG_BATCH]
        content = [text_part(
            f"文書: {doc_title}\nセクション: {topic}\n\n=== 本文 ===\n{body[:3000]}\n\n"
            f"以下の図 {len(batch)} 枚を、index 1〜{len(batch)} の順に説明してください。"
        )]
        for i, fig in enumerate(batch, start=1):
            content.append(text_part(f"図 index={i}（p{fig.page}）:"))
            content.append(image_part(fig.jpeg, detail="high"))  # 画面内の小さな文字を読ませる

        result = llm_json(FIGURE_SYSTEM, content)
        by_index = {int(f.get("index", 0)): f for f in result.get("figures", []) if isinstance(f, dict)}
        for i, fig in enumerate(batch, start=1):
            f = by_index.get(i)
            if f and f.get("keep", True) and str(f.get("description", "")).strip():
                kept.append((fig, str(f["description"]).strip()))
    return kept


# ========== 5. 出力 ==========

def sanitize_line(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def sanitize_body(body: str) -> str:
    # LLMが改行を「\n」という文字のまま出力することがあるため改行に戻す
    # （直前が英数字かつ直後が英字の場合は C:\new のようなパスの可能性があるので残す）
    body = re.sub(r"(?<=[^\x00-\x7F])\\n|\\n(?![A-Za-z])", "\n", body)
    # 既存の登録スクリプトは行頭の [FigN] / [caseN] で本文を区切るため、本文中の該当行を無効化する
    lines = []
    for line in body.splitlines():
        if re.match(r"\s*\[(Fig|case)\d", line):
            line = "※" + line.strip()
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def write_section(out_dir: Path, base_name: str, doc_title: str, structured: dict,
                  figures: list[tuple[Figure, str]], meta: dict):
    sec_dir = out_dir / base_name
    sec_dir.mkdir(parents=True, exist_ok=True)

    lines = [f"title[{sanitize_line(doc_title)}]",
             f"topic[{sanitize_line(structured.get('topic', ''))}]"]
    for item in structured.get("items", [])[:5]:
        if sanitize_line(item):
            lines.append(f"item[{sanitize_line(item)}]")
    lines += ["", sanitize_body(structured.get("body", "")), ""]

    fig_meta = []
    for n, (fig, desc) in enumerate(figures, start=1):
        fig_id = f"Fig{n}"
        (sec_dir / f"{fig_id}.jpg").write_bytes(fig.jpeg)
        lines += [f"[{fig_id}]", desc, ""]
        fig_meta.append({"id": fig_id, "page": fig.page, "bbox": fig.bbox})

    (sec_dir / f"{base_name}.txt").write_text("\n".join(lines), encoding="utf-8")
    (sec_dir / "meta.json").write_text(
        json.dumps({**meta, "base_name": base_name, "figures": fig_meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ========== メイン処理 ==========

def derive_doc_id(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_") or "doc"


def load_pages(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".txt", ".md"):
        return load_txt(path)
    raise ValueError(f"未対応のファイル形式です: {path.name}")


def print_dry_run(path: Path, pages: list[Page]):
    total_chars = sum(len(p.text) for p in pages)
    total_figs = sum(len(p.figures) for p in pages)
    print(f"\n📄 {path.name}: {len(pages)}ページ / {total_chars:,}文字 / 図 {total_figs}枚")
    for p in pages:
        flag = "  ← 文字が少ない（画像から読み取り対象）" if p.render and len(p.text.strip()) < OCR_MIN_CHARS else ""
        print(f"  p{p.no:>3}: {len(p.text):>5}文字, 図 {len(p.figures)}枚{flag}")


def process_file(path: Path, doc_id: str, max_sections: int | None, force: bool):
    out_dir = OUTPUT_DIR / doc_id
    if out_dir.exists() and not force:
        raise FileExistsError(
            f"{out_dir} は既に存在します。確認・修正済みの内容を上書きしないよう停止しました。"
            "上書きする場合は --force を付けてください。"
        )

    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"\n📄 {path.name} を処理します（doc_id={doc_id}）")
    pages = load_pages(path)
    print(f"  {len(pages)}ページ / 図 {sum(len(p.figures) for p in pages)}枚 を抽出")

    ocr_missing_text(pages)

    print("  🧭 セクション分割中...")
    outline = make_outline(pages, path.name)
    doc_title = outline.get("doc_title") or path.stem
    lang = outline.get("lang", "ja")
    sections = [s for s in outline["sections"] if not s["skip"]]
    skipped = [s for s in outline["sections"] if s["skip"]]
    print(f"  タイトル: {doc_title} / 言語: {lang} / セクション {len(sections)}件（除外 {len(skipped)}件）")
    if max_sections:
        sections = sections[:max_sections]
        print(f"  --max-sections により先頭 {len(sections)}件のみ処理します")

    page_by_no = {p.no: p for p in pages}

    def run(idx_sec):
        idx, sec = idx_sec
        base_name = f"{doc_id}_{idx:03d}"
        sec_pages = [page_by_no[n] for n in range(sec["start_page"], sec["end_page"] + 1)]
        try:
            structured = structure_section(doc_title, path.name, sec, sec_pages)
            figures = [f for p in sec_pages for f in p.figures]
            described = describe_figures(doc_title, structured.get("topic", sec["topic"]),
                                         structured.get("body", ""), figures) if figures else []
            meta = {
                "doc_id": doc_id,
                "source_file": path.name,
                "source_sha256": source_sha,
                "pages": [sec["start_page"], sec["end_page"]],
                "lang": "ja",
                "source_lang": lang,
                "original_text": "\n\n".join(p.text for p in sec_pages) if lang != "ja" else None,
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            write_section(out_dir, base_name, doc_title, structured, described, meta)
            print(f"  ✅ {base_name} (p{sec['start_page']}-{sec['end_page']}) "
                  f"{structured.get('topic', '')} / 図 {len(described)}/{len(figures)}枚採用")
            return {"base_name": base_name, "sec": sec, "topic": structured.get("topic", ""),
                    "chars": len(structured.get("body", "")), "figs": len(described),
                    "figs_total": len(figures), "error": None}
        except Exception as e:
            print(f"  ❌ {base_name} (p{sec['start_page']}-{sec['end_page']}): {e}")
            return {"base_name": base_name, "sec": sec, "topic": sec["topic"], "chars": 0,
                    "figs": 0, "figs_total": 0, "error": str(e)}

    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(run, enumerate(sections, start=1)))

    write_index(out_dir, path, doc_id, doc_title, lang, source_sha, results, skipped)
    errors = [r for r in results if r["error"]]
    print(f"  📋 確認用一覧: {out_dir / 'index.md'}")
    if errors:
        print(f"  ⚠ {len(errors)}件のセクションでエラーが発生しました（再実行するとそのセクションだけ処理されます）")


def write_index(out_dir, path, doc_id, doc_title, lang, source_sha, results, skipped):
    lines = [
        f"# {doc_title}",
        "",
        f"- 元ファイル: {path.name}",
        f"- doc_id: {doc_id} / 言語: {lang} / SHA256: {source_sha[:16]}…",
        f"- 生成日時: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "| base_name | ページ | トピック | 本文文字数 | 図（採用/候補） | 状態 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r["sec"]
        state = f"❌ {r['error']}" if r["error"] else "OK"
        lines.append(f"| {r['base_name']} | {s['start_page']}-{s['end_page']} | {r['topic']} | "
                     f"{r['chars']} | {r['figs']}/{r['figs_total']} | {state} |")
    if skipped:
        lines += ["", "## 除外したページ（表紙・目次など）", ""]
        lines += [f"- p{s['start_page']}-{s['end_page']}: {s['topic']}" for s in skipped]
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="PDF / txt からRAG登録用のタグ付きtxtと図を自動生成します")
    parser.add_argument("inputs", nargs="+", help="入力ファイル（PDF / txt / md）またはフォルダ")
    parser.add_argument("--doc-id", help="出力名の接頭辞（例: RAYFace_JP）。入力が1ファイルのときのみ指定可")
    parser.add_argument("--dry-run", action="store_true", help="APIを使わず抽出結果だけ表示")
    parser.add_argument("--max-sections", type=int, help="処理するセクション数の上限（お試し用）")
    parser.add_argument("--force", action="store_true", help="既存の出力フォルダを上書きする")
    args = parser.parse_args()

    files = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in (".pdf", ".txt", ".md"))
        elif p.exists():
            files.append(p)
        else:
            parser.error(f"見つかりません: {inp}")
    if args.doc_id and len(files) != 1:
        parser.error("--doc-id は入力が1ファイルのときのみ指定できます")

    for f in files:
        if args.dry_run:
            print_dry_run(f, load_pages(f))
        else:
            process_file(f, args.doc_id or derive_doc_id(f), args.max_sections, args.force)

    if not args.dry_run:
        cost = (_usage["input_tokens"] * PRICE_INPUT + _usage["output_tokens"] * PRICE_OUTPUT) / 1_000_000
        print(f"\n💰 API呼び出し {_usage['calls']}回（キャッシュ利用 {_usage['cache_hits']}回）"
              f" / 入力 {_usage['input_tokens']:,} / 出力 {_usage['output_tokens']:,} トークン"
              f" / 概算 ${cost:.2f}")


if __name__ == "__main__":
    main()
