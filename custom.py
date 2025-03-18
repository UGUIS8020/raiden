from typing import Optional
from langchain_core.tools import BaseTool
from langchain_community.tools.vectorstore.tool import BaseVectorStoreTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

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
        """Use the tool."""
        from langchain.chains.retrieval_qa.base import RetrievalQA

        # retrieverにkを渡す
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 13}  # ← ここを可変にしてもOK！
        )

        chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=retriever
        )

        return chain.invoke(
            {chain.input_key: query},
            config={"callbacks": run_manager.get_child() if run_manager else None},
        )[chain.output_key]

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        from langchain.chains.retrieval_qa.base import RetrievalQA

        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 13}
        )

        chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=retriever
        )

        return (
            await chain.ainvoke(
                {chain.input_key: query},
                config={"callbacks": run_manager.get_child() if run_manager else None},
            )
        )[chain.output_key]