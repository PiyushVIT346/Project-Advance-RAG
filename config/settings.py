from enum import Enum
from dataclasses import dataclass

class QueryType(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    SPECIFIC_ENTITY = "specific_entity"
    COMPLEX_REASONING = "complex_reasoning"
    SUMMARIZATION = "summarization"

class RetrievalStrategy(Enum):
    SEMANTIC_SIMILARITY = "semantic_similarity"
    BM25_KEYWORD = "bm25_keyword"
    HYBRID_ENSEMBLE = "hybrid_ensemble"
    RAG_FUSION = "rag_fusion"
    HYDE = "hyde"

class RerankingMethod(Enum):
    CROSS_ENCODER = "cross_encoder"
    COHERE_RERANK = "cohere_rerank"
    BGE_RERANK = "bge_rerank"
    CLUSTERING_FILTER = "clustering_filter"
    REDUNDANCY_FILTER = "redundancy_filter"

@dataclass
class RAGConfig:
    retrieval_strategy: RetrievalStrategy
    reranking_method: RerankingMethod
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 10
    rerank_top_k: int = 5
