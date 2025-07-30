from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def load_document(self, file_path: str, file_type: str):
        loaders = {
            'pdf': PyPDFLoader,
            'docx': Docx2txtLoader,
            'doc': Docx2txtLoader
        }
        if file_type not in loaders:
            raise ValueError(f"Unsupported file type: {file_type}")
        loader = loaders[file_type](file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
