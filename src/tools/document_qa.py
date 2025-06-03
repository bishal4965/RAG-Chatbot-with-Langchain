from langchain.tools import BaseTool


class DocumentQATool(BaseTool):
    """Tool for answering questions from documents"""
    name = "document_qa"
    description = "Answer questions based on uploaded documents"

    def __init__(self, qa_chain):
        super().__init__()
        self.qa_chain = qa_chain

    def _run(self, query: str) -> str:
        """Run document Q&A"""
        try:
            if self.qa_chain is None:
                return "No documents have been uploaded yet. Please upload documents first."
            
            result = self.qa_chain.run(query)
            return result
        except Exception as e:
            return f"Sorry I couldn't find relevant information in the documents: {str(e)}"