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
        try:
            if self.qa_chain is None:
                return "No documents have been uploaded yet. Please upload documents first."
            
            result = self.qa_chain.invoke(query)
            return result
        except Exception as e:
            return f"Sorry I couldn't find relevant information in the documents: {str(e)}"