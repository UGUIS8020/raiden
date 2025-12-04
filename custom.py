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

    def _get_vector_id(self, doc: Document) -> str:
        """ドキュメントからVector IDを取得（Qdrant対応）"""
        vector_id = (
            doc.metadata.get('vector_id') or 
            doc.metadata.get('id') or 
            doc.metadata.get('_id') or
            getattr(doc, 'id', None)
        )
        return str(vector_id) if vector_id else 'N/A'

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """ツールの実行メソッド"""
        from langchain.chains.retrieval_qa.base import RetrievalQA

        # ドキュメントを100件取得
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=100)
        
        # ★ デバッグ: 最初のドキュメントのメタデータを全て表示
        if len(docs_and_scores) > 0:
            # 最初の3件のVector IDを取得
            vector_ids = [self._get_vector_id(doc) for doc, _ in docs_and_scores[:3]]
            logger.info(f"📋 取得した最初の3件のVector ID: {vector_ids}")
            
            # Qdrant Clientで直接確認
            from qdrant_client import QdrantClient
            import os
            client = QdrantClient(
                url=os.getenv('QDRANT_URL'),
                api_key=os.getenv('QDRANT_API_KEY'),
            )
            
            # 最初のドキュメントのpayloadを直接取得
            point = client.retrieve(
                collection_name="raiden-main",
                ids=[vector_ids[0]],
                with_payload=True
            )
            logger.info(f"✅ 直接取得したpayload: {point[0].payload}")
        
        logger.debug(f"📥 {len(docs_and_scores)}件のドキュメントを取得")
        logger.debug("--- 重み付け前のドキュメント（全100件） ---")
        for i, (doc, score) in enumerate(docs_and_scores):
            vector_id = self._get_vector_id(doc)
            weight = doc.metadata.get('weight', 1.0)
            doc_type = doc.metadata.get('type', 'N/A')  # ← これが取得できていない
            logger.debug(
                f"Doc {i}: Score={score:.6f}, Weight={weight}, "
                f"Type={doc_type}, VectorID={vector_id}"
            )

        # 重みによる再ランキング
        weighted_docs = []
        for doc, score in docs_and_scores:
            weight = doc.metadata.get("weight", 1.0)
            weighted_score = score * weight
            doc.metadata["original_score"] = score
            doc.metadata["weighted_score"] = weighted_score
            weighted_docs.append((doc, weighted_score))

        # 重み付けされたスコアでソート（降順）
        sorted_docs = sorted(weighted_docs, key=lambda x: x[1], reverse=True)

        # 上位15件のドキュメントを使用
        top_docs = [doc for doc, _ in sorted_docs[:15]]

        # 重要な情報はINFOレベルで出力
        logger.info("🔄 重み付け再ランキング完了")
        logger.info(f"📊 上位3件のタイプ: {[doc.metadata.get('type', 'N/A') for doc in top_docs[:3]]}")
        
        # 詳細ログ
        logger.debug("--- 重み付け後のドキュメント（上位15件） ---")
        for i, doc in enumerate(top_docs):
            vector_id = self._get_vector_id(doc)
            weighted = doc.metadata.get('weighted_score', 'N/A')
            original = doc.metadata.get('original_score', 'N/A')
            weight = doc.metadata.get('weight', 1.0)
            doc_type = doc.metadata.get('type', 'N/A')
            logger.debug(
                f"Doc {i}: Weighted={weighted:.6f}, Original={original:.6f}, "
                f"Weight={weight}, Type={doc_type}, VectorID={vector_id}"
            )

        # カスタムリトリーバーに渡す
        retriever = StaticDocRetriever(top_docs)

        # QA チェーン作成
        chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=retriever,
            verbose=False
        )

        # 実行
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