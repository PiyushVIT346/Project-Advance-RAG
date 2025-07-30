import re
import google.generativeai as genai
from config.settings import QueryType, RAGConfig, RetrievalStrategy, RerankingMethod

class QueryAnalyzer:
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.patterns = {
            'factual': [
                r'\b(what|who|where|when|which|how many)\b',
                r'\b(define|definition|meaning)\b',
                r'\b(is|are|was|were)\b.*\?'
            ],
            'analytical': [
                r'\b(why|how|explain|analyze|compare|contrast)\b',
                r'\b(relationship|correlation|impact|effect)\b',
                r'\b(cause|reason|factor)\b'
            ],
            'temporal': [
                r'\b(before|after|during|timeline|chronology)\b',
                r'\b(first|last|previous|next|recent)\b',
                r'\b(year|month|day|date|time)\b'
            ]
        }

    def categorize_query(self, query: str) -> QueryType:
        q = query.lower()
        if any(re.search(p, q) for p in self.patterns['factual']):
            return QueryType.FACTUAL
        elif any(re.search(p, q) for p in self.patterns['analytical']):
            return QueryType.ANALYTICAL
        elif any(re.search(p, q) for p in self.patterns['temporal']):
            return QueryType.TEMPORAL
        elif len(query.split()) > 15:
            return QueryType.COMPLEX_REASONING
        elif 'compare' in q or 'versus' in q:
            return QueryType.COMPARATIVE
        elif 'summary' in q or 'summarize' in q:
            return QueryType.SUMMARIZATION
        else:
            return QueryType.SPECIFIC_ENTITY

    def determine_strategy(self, query: str, doc_count: int) -> RAGConfig:
        query_type = self.categorize_query(query)
        strategy_configs = {
            QueryType.FACTUAL: RAGConfig(
                retrieval_strategy=RetrievalStrategy.BM25_KEYWORD,
                reranking_method=RerankingMethod.CROSS_ENCODER,
                top_k=5, rerank_top_k=3
            ),
            QueryType.ANALYTICAL: RAGConfig(
                retrieval_strategy=RetrievalStrategy.HYBRID_ENSEMBLE,
                reranking_method=RerankingMethod.COHERE_RERANK,
                top_k=8, rerank_top_k=5
            ),
            QueryType.COMPARATIVE: RAGConfig(
                retrieval_strategy=RetrievalStrategy.RAG_FUSION,
                reranking_method=RerankingMethod.CLUSTERING_FILTER,
                top_k=12, rerank_top_k=6
            ),
            QueryType.TEMPORAL: RAGConfig(
                retrieval_strategy=RetrievalStrategy.SEMANTIC_SIMILARITY,
                reranking_method=RerankingMethod.CROSS_ENCODER,
                top_k=10, rerank_top_k=5
            ),
            QueryType.COMPLEX_REASONING: RAGConfig(
                retrieval_strategy=RetrievalStrategy.HYDE,
                reranking_method=RerankingMethod.BGE_RERANK,
                top_k=15, rerank_top_k=8
            ),
            QueryType.SUMMARIZATION: RAGConfig(
                retrieval_strategy=RetrievalStrategy.SEMANTIC_SIMILARITY,
                reranking_method=RerankingMethod.REDUNDANCY_FILTER,
                top_k=20, rerank_top_k=10
            ),
            QueryType.SPECIFIC_ENTITY: RAGConfig(
                retrieval_strategy=RetrievalStrategy.SEMANTIC_SIMILARITY,
                reranking_method=RerankingMethod.CROSS_ENCODER,
                top_k=6, rerank_top_k=3
            )
        }
        return strategy_configs.get(query_type, strategy_configs[QueryType.FACTUAL])
