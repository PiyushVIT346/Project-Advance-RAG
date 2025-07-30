from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from logging_config.logger import logger
import re

class AdvancedRetriever:
    def __init__(self, documents, embedding_manager):
        self.documents = documents
        self.embedding_manager = embedding_manager
        self.vector_store = None
        self.bm25_retriever = None
        self._setup_retrievers()

    def _setup_retrievers(self):
        try:
            embeddings = self.embedding_manager.get_embeddings()
            self.vector_store = FAISS.from_documents(self.documents, embeddings)
            texts = [doc.page_content for doc in self.documents]
            self.bm25_retriever = BM25Retriever.from_texts(texts)
            logger.info(f"Initialized retrievers with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Failed to setup retrievers: {e}")
            raise

    def semantic_similarity_retrieval(self, query, k=10):
        return self.vector_store.similarity_search(query, k=k)

    def bm25_retrieval(self, query, k=10):
        self.bm25_retriever.k = k
        return self.bm25_retriever.invoke(query)

    def hybrid_ensemble_retrieval(self, query, k=10):
        vector_retriever = self.vector_store.as_retriever(search_kwargs={'k': k//2})
        self.bm25_retriever.k = k//2
        ensemble = EnsembleRetriever([vector_retriever, self.bm25_retriever], weights=[0.6, 0.4])
        return ensemble.get_relevant_documents(query)

    def rag_fusion_retrieval(self, query, k=10, gemini_model=None):
        if not gemini_model:
            return self.semantic_similarity_retrieval(query, k)
        try:
            prompt = f"Generate 3 different variations of this query:\n{query}\n\nFormat as:\n1. [variation]\n2. [variation]\n3. [variation]"
            response = gemini_model.generate_content(prompt)
            variations = [query]
            for line in response.text.split('\n'):
                if re.match(r'^\d+\.', line.strip()):
                    variation = re.sub(r'^\d+\.\s*', '', line.strip())
                    if variation and variation not in variations:
                        variations.append(variation)
            all_docs = []
            docs_per_variation = max(1, k // len(variations))
            for variation in variations[:4]:
                docs = self.semantic_similarity_retrieval(variation, docs_per_variation)
                all_docs.extend(docs)
            unique_docs, seen_content = [], set()
            for doc in all_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)
            return unique_docs[:k]
        except Exception as e:
            logger.warning(f"RAG fusion failed: {e}")
            return self.semantic_similarity_retrieval(query, k)

    def hyde_retrieval(self, query, k=10, gemini_model=None):
        if not gemini_model:
            return self.semantic_similarity_retrieval(query, k)
        try:
            prompt = f"Write a hypothetical document paragraph that would perfectly answer this question: {query}"
            response = gemini_model.generate_content(prompt)
            return self.vector_store.similarity_search(response.text, k=k)
        except Exception as e:
            logger.warning(f"HyDE retrieval failed: {e}")
            return self.semantic_similarity_retrieval(query, k)
