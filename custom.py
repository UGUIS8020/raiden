from typing import Optional, List
from langchain_core.tools import BaseTool
from langchain_community.tools.vectorstore.tool import BaseVectorStoreTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.schema import BaseRetriever
from langchain_core.documents import Document

class StaticDocRetriever(BaseRetriever):
    def __init__(self, documents: List[Document]):
        super().__init__()
        self.__dict__["documents"] = documents  # 強制的に属性として登録

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents

class CustomVectorStoreQATool(BaseVectorStoreTool, BaseTool):
    """Tool for the VectorDBQA chain. To be initialized with name and chain."""

    @staticmethod
    def get_description(name: str, description: str) -> str:
        template: str = (
            "Useful for when you need to answer questions about {name}. "
            "Whenever you need information about {description} "
            "you should ALWAYS use this. "
            "Input should be a fully formed question."
        )
        return template.format(name=name, description=description)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from langchain.chains.retrieval_qa.base import RetrievalQA

        # ドキュメントを20件取得
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=100)

        print("--- 重み付け前のドキュメント ---")
        for i, (doc, score) in enumerate(docs_and_scores):
            print(f"Doc {i}: Score={score}, Weight={doc.metadata.get('weight', 1.0)}, Vector ID={doc.metadata.get('vector_id', 'N/A')}")

        # 重みによる再ランキング
        weighted_docs = []
        for doc, score in docs_and_scores:
            weight = doc.metadata.get("weight", 1.0)
            weighted_score = score * weight  # または score / weight
            doc.metadata["original_score"] = score
            doc.metadata["weighted_score"] = weighted_score
            weighted_docs.append((doc, weighted_score))

        # 重み付けされたスコアでソート
        sorted_docs = sorted(weighted_docs, key=lambda x: x[1], reverse=True)

        # 上位5件のドキュメントを使用
        top_docs = [doc for doc, _ in sorted_docs[:15]]

        print("--- 重み付け後のドキュメント ---")
        for i, doc in enumerate(top_docs):
            print(f"Weighted={doc.metadata.get('weighted_score', 'N/A')}, "
                  f"Doc {i}: Original={doc.metadata.get('original_score', 'N/A')}, "
                  f"Weight={doc.metadata.get('weight', 1.0)}, "  
                  f"Doc {i}: Vector ID={doc.metadata.get('vector_id', 'N/A')}, "                  
            )
                  

        # カスタムリトリーバーに渡す
        retriever = StaticDocRetriever(top_docs)

        # QA チェーン作成
        chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=retriever
        )

        # 実行
        return chain.invoke(
            {chain.input_key: query},
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )[chain.output_key]