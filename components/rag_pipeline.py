import google.generativeai as genai
from components.query_analyzer import QueryAnalyzer
from components.document_processor import DocumentProcessor
from components.embeddings import EmbeddingManager
from components.retriever import AdvancedRetriever
from components.reranker import AdvancedReranker
from config.settings import RetrievalStrategy, RerankingMethod
from logging_config.logger import logger

class AdaptiveRAGSystem:
    def __init__(self, gemini_api_key: str, cohere_api_key=None):
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        self.query_analyzer = QueryAnalyzer(gemini_api_key)
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.documents = []
        self.retriever = None
        self.reranker = AdvancedReranker(cohere_api_key)

    def load_documents(self, file_path: str, file_type: str):
        try:
            self.documents = self.doc_processor.load_document(file_path, file_type)
            self.retriever = AdvancedRetriever(self.documents, self.embedding_manager)
            logger.info(f"Successfully loaded {len(self.documents)} document chunks")
        except Exception as e:
            logger.error(f"Document loading failed: {e}")
            raise

    def retrieve_documents(self, query: str, config):
        if not self.retriever:
            raise ValueError("No documents loaded")
        strategy_methods = {
            RetrievalStrategy.SEMANTIC_SIMILARITY: self.retriever.semantic_similarity_retrieval,
            RetrievalStrategy.BM25_KEYWORD: self.retriever.bm25_retrieval,
            RetrievalStrategy.HYBRID_ENSEMBLE: self.retriever.hybrid_ensemble_retrieval,
            RetrievalStrategy.RAG_FUSION: lambda q, k: self.retriever.rag_fusion_retrieval(q, k, self.gemini_model),
            RetrievalStrategy.HYDE: lambda q, k: self.retriever.hyde_retrieval(q, k, self.gemini_model),
        }
        method = strategy_methods.get(config.retrieval_strategy, self.retriever.semantic_similarity_retrieval)
        return method(query, config.top_k)

    def rerank_documents(self, query: str, docs, config):
        if not docs or not self.reranker:
            return docs[:config.rerank_top_k]
        rerank_methods = {
            RerankingMethod.CROSS_ENCODER: self.reranker.cross_encoder_rerank,
            RerankingMethod.COHERE_RERANK: self.reranker.cohere_rerank,
            RerankingMethod.BGE_RERANK: self.reranker.bge_rerank,
            RerankingMethod.CLUSTERING_FILTER: self.reranker.clustering_filter,
            RerankingMethod.REDUNDANCY_FILTER: self.reranker.redundancy_filter,
        }
        method = rerank_methods.get(config.reranking_method, self.reranker.cross_encoder_rerank)
        return method(query, docs, config.rerank_top_k)

    def generate_answer(self, query, context_docs):
        if not context_docs:
            return "I couldn't find relevant information to answer your question."
        context = "\n\n".join([doc.page_content for doc in context_docs])
        prompt = f"""Based on the following context, answer the user's question comprehensively and accurately.

Context:
{context}

Question: {query}

Instructions:
- Provide a detailed, well-structured answer
- Use only information from the provided context
- If the context doesn't contain enough information, state this clearly
- Include relevant quotes or specific details when appropriate
- Structure your response clearly with appropriate formatting

Answer:"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"

    def query(self, question: str):
        if not self.documents:
            return {"error": "No documents loaded"}
        try:
            config = self.query_analyzer.determine_strategy(question, len(self.documents))
            query_type = self.query_analyzer.categorize_query(question)
            retrieved_docs = self.retrieve_documents(question, config)
            final_docs = self.rerank_documents(question, retrieved_docs, config)
            answer = self.generate_answer(question, final_docs)
            return {
                "answer": answer,
                "query_type": query_type.value,
                "retrieval_strategy": config.retrieval_strategy.value,
                "reranking_method": config.reranking_method.value,
                "num_retrieved": len(retrieved_docs),
                "num_final": len(final_docs),
                "retrieved_documents": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "metadata": doc.metadata
                    } for doc in final_docs
                ]
            }
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {"error": f"Query processing failed: {str(e)}"}
