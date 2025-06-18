from langchain.tools import BaseTool
from pydantic import Field
from typing import Any

class DocumentQATool(BaseTool):
    """Tool for answering questions from documents"""
    name: str = "document_qa"
    description: str = "Answer questions based on uploaded documents"
    qa_chain: Any = Field(default=None, description="The QA chain for document processing")

    def __init__(self, qa_chain):
        super().__init__()
        self.qa_chain = qa_chain

    def _run(self, query: str) -> str:
        """Run document Q&A"""

        # print(f"QA chain available: {self.qa_chain is not None}")
        try:
            if self.qa_chain is None:
                return "No documents have been uploaded yet. Please upload documents first."
            
            # Test if vector store has content
            if hasattr(self.qa_chain, 'retriever'):
                docs = self.qa_chain.retriever.get_relevant_documents(query)
                print(f"Retrieved {len(docs)} relevant documents")
            
            
            result = self.qa_chain.invoke(query)

            # Extract the actual answer from the result
            if isinstance(result, dict):
                return result.get('answer', result.get('result', str(result)))
            
            return str(result)
        except Exception as e:
            return f"Sorry I couldn't find relevant information in the documents: {str(e)}"