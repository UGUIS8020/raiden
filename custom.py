from typing import Optional, List
from langchain_core.tools import BaseTool
from langchain_community.tools.vectorstore.tool import BaseVectorStoreTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.schema import BaseRetriever
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

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
        """ドキュメントから Qdrant の point.id を推定"""
        vector_id = (
            doc.metadata.get('id')           # ← まず metadata['id']（UUID）を使う
            or doc.metadata.get('_id')
            or getattr(doc, 'id', None)
            or doc.metadata.get('vector_id') # ← どうしても無ければ最後に vector_id
        )
        return str(vector_id) if vector_id is not None else None

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """ツールの実行メソッド"""
        from langchain.chains.retrieval_qa.base import RetrievalQA
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        TOP_K = 15
        FIGURE_K = 5  # 図の説明文は本文とは別枠で取得（件数が多く本文を押し出さないように）

        def search(types, k):
            return self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=Filter(must=[FieldCondition(key="type", match=MatchAny(any=types))]),
            )

        # ★ 本文・論文の上位15件 + 図の説明文の上位5件
        docs_and_scores = (
            search(["content", "pubmed_paper"], TOP_K)
            + search(["figure_description"], FIGURE_K)
        )

        logger.debug(f"📥 {len(docs_and_scores)}件のドキュメントを取得")

        # 類似文書がゼロのときの保険
        if not docs_and_scores:
            logger.warning("⚠ 類似ドキュメントが1件も見つかりませんでした")
            return "関連する文書が見つかりませんでした。別の聞き方で質問してみてください。"

        top_docs_with_scores = docs_and_scores  # [(doc, score), ...] 本文最大15件 + 図最大5件

        # ★ 取得した文書だけpayloadを取得（高速化）
        # VectorStoreが保持しているクライアントを再利用（毎回の接続を避ける）
        client = self.vectorstore.client

        raw_vector_ids = [self._get_vector_id(doc) for doc, _ in top_docs_with_scores]
        vector_ids = [vid for vid in raw_vector_ids if vid is not None]

        points = client.retrieve(
            collection_name=self.vectorstore.collection_name,
            ids=vector_ids,
            with_payload=True
        )

        payload_map = {str(point.id): point.payload for point in points}

        top_docs = []
        for doc, score in top_docs_with_scores:
            vector_id = self._get_vector_id(doc)
            if vector_id and vector_id in payload_map:
                payload = payload_map[vector_id] or {}

                # ✅ payload → metadata にマージ（text は除外）
                for key, value in payload.items():
                    if key == "text":
                        continue
                    doc.metadata[key] = value

                # original_id が無い古いデータ用のフォールバック
                if "original_id" not in doc.metadata:
                    doc.metadata["original_id"] = (
                        doc.metadata.get("vector_id") or str(vector_id)
                    )

            # スコアは常に付与
            doc.metadata["original_score"] = score
            top_docs.append(doc)

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
            retriever=retriever,
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