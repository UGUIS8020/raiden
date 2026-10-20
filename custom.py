from typing import Optional, List
from langchain_core.tools import BaseTool
from langchain_community.tools.vectorstore.tool import BaseVectorStoreTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.schema import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from prompts import QA_PROMPT_TEMPLATE
import logging

logger = logging.getLogger(__name__)

QA_PROMPT = PromptTemplate(template=QA_PROMPT_TEMPLATE, input_variables=["context", "question"])


def source_label(meta: dict) -> str:
    """回答で出典を示せるよう、文書の先頭に付ける出典表記"""
    kind = "（図）" if meta.get("type") == "figure_description" else ""
    if meta.get("source_file"):
        pages = meta.get("pages") or []
        page = ""
        if len(pages) == 2:
            page = f" p.{pages[0]}" if pages[0] == pages[1] else f" p.{pages[0]}-{pages[1]}"
        return f"【出典: {meta['source_file']}{page}{kind}】"
    if meta.get("pmid"):
        return f"【出典: PubMed {meta['pmid']} {meta.get('title', '')}】"
    if meta.get("title"):
        return f"【出典: {meta['title']}{kind}】"
    return ""


import threading

# 直近の検索で回答に渡した資料（Q&Aログに記録するため。リクエストごとのスレッド単位で保持）
_last = threading.local()


def clear_last_sources():
    _last.sources = []


def pop_last_sources() -> list:
    sources = getattr(_last, "sources", [])
    _last.sources = []
    return sources


QUERY_REWRITE_PROMPT = """あなたは歯科医療・歯科機器（RAYFace など）の資料検索のための検索クエリ作成者です。
ユーザーの質問から、資料の該当箇所を見つけるための検索クエリを2つ作ってください。
- 質問の核心（何について知りたいか）だけを表す。質問者が示した前提や候補は、どちらのクエリにも含めない。
  例: 「Aの設定はBとCでいいですか？」→ B・C は含めず「Aの設定方法」「A 設定 位置」のようにする。
- 1つ目: 質問の核心を短い文で。2つ目: 資料で使われていそうな機能名・画面名・専門用語のキーワードの並び。
出力JSON: {{"queries": ["...", "..."]}}

質問: {question}"""


def rewrite_queries(llm, query: str) -> List[str]:
    """誘導的な語に検索が引っ張られないよう、質問の核心を表す検索文を追加する（失敗時は元の質問のみ）"""
    import json, re
    try:
        text = llm.invoke(QUERY_REWRITE_PROMPT.format(question=query)).content
        m = re.search(r"\{.*\}", text, re.S)
        extra = [q for q in json.loads(m.group(0)).get("queries", []) if isinstance(q, str) and q.strip()] if m else []
    except Exception as e:
        logger.warning(f"検索クエリの作成に失敗: {e}")
        extra = []
    return [query] + extra[:2]


TOP_K = 15
FIGURE_K = 5  # 図の説明文は本文とは別枠で取得（件数が多く本文を押し出さないように）


def doc_point_id(doc: Document) -> Optional[str]:
    """ドキュメントから Qdrant の point.id を推定"""
    vector_id = (
        doc.metadata.get('id')            # まず metadata['id']（UUID）
        or doc.metadata.get('_id')
        or getattr(doc, 'id', None)
        or doc.metadata.get('vector_id')  # どうしても無ければ vector_id
    )
    return str(vector_id) if vector_id is not None else None


def fused_search(vectorstore, queries: List[str], types: List[str], k: int):
    """複数の検索文で検索し、順位の融合（RRF）でまとめる。
    類似度の値で並べると、元の質問の誘導的な語で上位に来た文書に、
    言い換えた検索文で見つかった文書が押し出されるため（eval occ-07 で確認）"""
    from qdrant_client.models import Filter, FieldCondition, MatchAny
    fused, best = {}, {}
    for q in queries:
        results = vectorstore.similarity_search_with_score(
            q, k=k, filter=Filter(must=[FieldCondition(key="type", match=MatchAny(any=types))]),
        )
        for rank, (doc, score) in enumerate(results, start=1):
            key = doc_point_id(doc) or doc.page_content[:100]
            fused[key] = fused.get(key, 0.0) + 1.0 / (60 + rank)
            if key not in best or score > best[key][1]:
                best[key] = (doc, score)
    order = sorted(fused, key=fused.get, reverse=True)[:k]
    return [best[key] for key in order]


def retrieve_documents(vectorstore, llm, query: str) -> List[Document]:
    """回答に渡す資料を取得する（検索文の追加 → 融合検索 → payload付与 → セクション展開）。
    ツール（回答時）と eval/review_logs.py（答え合わせ時）で共通に使う"""
    queries = rewrite_queries(llm, query)
    logger.info(f"🔎 検索クエリ: {queries}")
    docs_and_scores = (
        fused_search(vectorstore, queries, ["content", "pubmed_paper"], TOP_K)
        + fused_search(vectorstore, queries, ["figure_description"], FIGURE_K)
    )
    if not docs_and_scores:
        return []

    # 取得した文書だけ payload を取得し、metadata にマージ（text は除外）
    client = vectorstore.client
    ids = [pid for pid in (doc_point_id(d) for d, _ in docs_and_scores) if pid]
    payload_map = {str(p.id): p.payload for p in client.retrieve(vectorstore.collection_name, ids=ids, with_payload=True)}
    docs = []
    for doc, score in docs_and_scores:
        pid = doc_point_id(doc)
        for key, value in (payload_map.get(pid) or {}).items():
            if key != "text":
                doc.metadata[key] = value
        doc.metadata.setdefault("original_id", doc.metadata.get("vector_id") or pid)
        doc.metadata["original_score"] = score
        docs.append(doc)

    # 小さいチャンクで見つけ、回答にはセクション全体を渡す（手順が途中で切れないように）
    return expand_sections(client, vectorstore.collection_name, docs)


MAX_SECTIONS = 6          # 回答に渡すセクション（展開後）の最大数
_section_cache: dict = {}  # base_name -> セクション全文（データ更新時はアプリ再起動で反映）


def _merge_chunks(chunks: List[str]) -> str:
    """チャンク分割時の重なり（overlap）を取り除いてつなげる"""
    merged = chunks[0]
    for nxt in chunks[1:]:
        overlap = next((k for k in range(min(len(merged), len(nxt), 200), 10, -1)
                        if merged.endswith(nxt[:k])), 0)
        merged += ("" if overlap else "\n") + nxt[overlap:]
    return merged


def section_text(client, collection: str, base_name: str) -> Optional[str]:
    """同じセクション（base_name）の本文チャンクをすべて取得し、1つの文章に戻す"""
    if base_name in _section_cache:
        return _section_cache[base_name]
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[
            FieldCondition(key="base_name", match=MatchValue(value=base_name)),
            FieldCondition(key="type", match=MatchValue(value="content")),
        ]),
        limit=100, with_payload=["text", "chunk_index"], with_vectors=False,
    )
    if not points:
        return None
    points.sort(key=lambda p: p.payload.get("chunk_index", 0))
    texts = [p.payload.get("text", "") for p in points]
    header_end = texts[0].find("\n\n") + 2 if texts[0].startswith("title[") else 0
    header = texts[0][:header_end]
    bodies = [t[header_end:] if header and t.startswith(header) else t for t in texts]
    text = header + _merge_chunks(bodies)
    _section_cache[base_name] = text
    return text


def expand_sections(client, collection: str, docs: List[Document]) -> List[Document]:
    """複数チャンクに分かれたセクションは、ヒットしたチャンクをセクション全体に置き換える"""
    result, expanded = [], []
    for doc in docs:
        meta = doc.metadata
        base = meta.get("base_name")
        is_chunked = meta.get("type") == "content" and base and (meta.get("chunk_count") or 1) > 1
        if is_chunked:
            if base in expanded or len(expanded) >= MAX_SECTIONS:
                continue  # 同じセクションの別チャンク、または上限超過
            try:
                full = section_text(client, collection, base)
            except Exception as e:
                logger.warning(f"セクション展開に失敗（{base}）: {e}")
                full = None
            if full:
                doc.page_content = full
                expanded.append(base)
        elif meta.get("type") == "figure_description" and base in expanded:
            continue  # 展開したセクション本文に図の説明が含まれている
        result.append(doc)
    return result

class StaticDocRetriever(BaseRetriever):
    """静的なドキュメントリストを返すRetriever"""
    
    def __init__(self, documents: List[Document]):
        super().__init__()
        self.__dict__["documents"] = documents

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents


class CustomVectorStoreQATool(BaseVectorStoreTool, BaseTool):
    """カスタムVectorStore QAツール（重み付け対応）"""

    @staticmethod
    def get_description(name: str, description: str) -> str:
        template: str = (
            "Useful for when you need to answer questions about {name}. "
            "Whenever you need information about {description} "
            "you should ALWAYS use this. "
            "Input should be a fully formed question."
        )
        return template.format(name=name, description=description)

    def _get_vector_id(self, doc: Document) -> Optional[str]:
        return doc_point_id(doc)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """ツールの実行メソッド"""
        from langchain.chains.retrieval_qa.base import RetrievalQA

        top_docs = retrieve_documents(self.vectorstore, self.llm, query)

        # 類似文書がゼロのときの保険
        if not top_docs:
            logger.warning("⚠ 類似ドキュメントが1件も見つかりませんでした")
            return "関連する文書が見つかりませんでした。別の聞き方で質問してみてください。"
        _last.sources = getattr(_last, "sources", []) + [{
            "point_id": doc_point_id(d),
            "vector_id": d.metadata.get("vector_id") or d.metadata.get("original_id"),
            "base_name": d.metadata.get("base_name"),
            "type": d.metadata.get("type"),
            "score": d.metadata.get("original_score"),
        } for d in top_docs]
        for doc in top_docs:
            label = source_label(doc.metadata)
            if label:
                doc.page_content = f"{label}\n{doc.page_content}"

        logger.info("🔄 検索完了")

        # ログ用（ID＋スコア）
        lines = []
        for i, doc in enumerate(top_docs, start=1):
            meta = getattr(doc, "metadata", {}) or {}

            vector_id  = meta.get("vector_id") or meta.get("original_id", "N/A")
            section    = meta.get("section", "N/A")
            lang       = meta.get("lang", "N/A")
            title      = meta.get("title", "") or ""

            score = meta.get("original_score", None)
            if not isinstance(score, (int, float)) and hasattr(doc, "score"):
                score = doc.score

            if isinstance(score, (int, float)):
                lines.append(
                    f"{i:2d}. {vector_id} "
                    f"[section={section}, lang={lang}] "
                    f"{title[:30]}... (score={score:.4f})"
                )
            else:
                lines.append(
                    f"{i:2d}. {vector_id} "
                    f"[section={section}, lang={lang}] "
                    f"{title[:30]}..."
                )

        log_text = "📊 上位{}件:\n{}".format(len(lines), "\n".join(lines))
        logger.info(log_text)

        retriever = StaticDocRetriever(top_docs)

        chain = RetrievalQA.from_chain_type(
            self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_PROMPT},
            verbose=False
        )

        try:
            result = chain.invoke(
                {chain.input_key: query},
                config={"callbacks": run_manager.get_child() if run_manager else None},
            )
            logger.info("✅ QAチェーン実行完了")
            return result[chain.output_key]
        except Exception as e:
            logger.error(f"❌ QAチェーン実行エラー: {e}", exc_info=True)
            raise