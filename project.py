import os
import streamlit as st
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import tempfile
import re
import warnings

# LangChain imports
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# ML/NLP imports
from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Optional imports
try:
    import cohere
except ImportError:
    cohere = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==== Configuration Classes ====

class QueryType(Enum):
    """Query categorization for adaptive retrieval."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    SPECIFIC_ENTITY = "specific_entity"
    COMPLEX_REASONING = "complex_reasoning"
    SUMMARIZATION = "summarization"

class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    BM25_KEYWORD = "bm25_keyword"
    HYBRID_ENSEMBLE = "hybrid_ensemble"
    RAG_FUSION = "rag_fusion"
    HYDE = "hyde"

class RerankingMethod(Enum):
    """Available reranking methods."""
    CROSS_ENCODER = "cross_encoder"
    COHERE_RERANK = "cohere_rerank"
    BGE_RERANK = "bge_rerank"
    CLUSTERING_FILTER = "clustering_filter"
    REDUNDANCY_FILTER = "redundancy_filter"

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    retrieval_strategy: RetrievalStrategy
    reranking_method: RerankingMethod
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 10
    rerank_top_k: int = 5

# ==== Core Components ====

class QueryAnalyzer:
    """Analyzes queries to determine optimal RAG configuration."""
    
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Query pattern matching
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
        """Categorize query based on patterns and content analysis."""
        q = query.lower()
        
        # Pattern-based categorization
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
        """Determine optimal RAG configuration based on query type."""
        query_type = self.categorize_query(query)
        
        # Strategy mapping
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

class DocumentProcessor:
    """Handles document loading and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_document(self, file_path: str, file_type: str) -> List[Document]:
        """Load and chunk document based on file type."""
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

class EmbeddingManager:
    """Manages embedding models with fallback options."""
    
    def __init__(self):
        self.embeddings = {}
        self.current_model = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding models with error handling."""
        models_to_try = [
            ("sentence_transformer", "all-MiniLM-L6-v2"),
            ("e5", "intfloat/e5-base-v2"),
            ("bge", "BAAI/bge-base-en-v1.5")
        ]
        
        for name, model_name in models_to_try:
            try:
                embedding = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                # Test the embedding
                _ = embedding.embed_query("test")
                self.embeddings[name] = embedding
                if self.current_model is None:
                    self.current_model = name
                logger.info(f"Successfully loaded embedding model: {name}")
            except Exception as e:
                logger.warning(f"Failed to load {name} ({model_name}): {e}")
                continue
        
        if not self.embeddings:
            raise RuntimeError("No embedding models could be loaded")
    
    def get_embeddings(self, model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
        """Get embedding model, fallback to available model if specified one fails."""
        model = model_name or self.current_model
        return self.embeddings.get(model, list(self.embeddings.values())[0])

class AdvancedRetriever:
    """Implements multiple retrieval strategies."""
    
    def __init__(self, documents: List[Document], embedding_manager: EmbeddingManager):
        self.documents = documents
        self.embedding_manager = embedding_manager
        self.vector_store = None
        self.bm25_retriever = None
        self._setup_retrievers()
    
    def _setup_retrievers(self):
        """Initialize vector store and BM25 retriever."""
        try:
            embeddings = self.embedding_manager.get_embeddings()
            self.vector_store = FAISS.from_documents(self.documents, embeddings)
            
            texts = [doc.page_content for doc in self.documents]
            self.bm25_retriever = BM25Retriever.from_texts(texts)
            
            logger.info(f"Initialized retrievers with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Failed to setup retrievers: {e}")
            raise
    
    def semantic_similarity_retrieval(self, query: str, k: int = 10) -> List[Document]:
        """Retrieve documents using semantic similarity."""
        return self.vector_store.similarity_search(query, k=k)
    
    def bm25_retrieval(self, query: str, k: int = 10) -> List[Document]:
        """Retrieve documents using BM25 keyword search."""
        self.bm25_retriever.k = k
        return self.bm25_retriever.get_relevant_documents(query)
    
    def hybrid_ensemble_retrieval(self, query: str, k: int = 10) -> List[Document]:
        """Combine semantic and keyword retrieval."""
        vector_retriever = self.vector_store.as_retriever(search_kwargs={'k': k//2})
        self.bm25_retriever.k = k//2
        
        ensemble = EnsembleRetriever([vector_retriever, self.bm25_retriever], weights=[0.6, 0.4])
        return ensemble.get_relevant_documents(query)
    
    def rag_fusion_retrieval(self, query: str, k: int = 10, gemini_model=None) -> List[Document]:
        """Generate query variations for comprehensive retrieval."""
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
            
            # Retrieve for each variation
            all_docs = []
            docs_per_variation = max(1, k // len(variations))
            
            for variation in variations[:4]:  # Limit to 4 variations
                docs = self.semantic_similarity_retrieval(variation, docs_per_variation)
                all_docs.extend(docs)
            
            # Remove duplicates based on content
            unique_docs = []
            seen_content = set()
            for doc in all_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)
            
            return unique_docs[:k]
            
        except Exception as e:
            logger.warning(f"RAG fusion failed: {e}")
            return self.semantic_similarity_retrieval(query, k)
    
    def hyde_retrieval(self, query: str, k: int = 10, gemini_model=None) -> List[Document]:
        """Use hypothetical document embedding for retrieval."""
        if not gemini_model:
            return self.semantic_similarity_retrieval(query, k)
        
        try:
            prompt = f"Write a hypothetical document paragraph that would perfectly answer this question: {query}"
            response = gemini_model.generate_content(prompt)
            return self.vector_store.similarity_search(response.text, k=k)
        except Exception as e:
            logger.warning(f"HyDE retrieval failed: {e}")
            return self.semantic_similarity_retrieval(query, k)

class AdvancedReranker:
    """Implements multiple reranking strategies."""
    
    def __init__(self, cohere_api_key: Optional[str] = None):
        self.cross_encoder = None
        self.bge_reranker = None
        self.cohere_client = None
        self.sentence_transformer = None
        
        self._initialize_rerankers(cohere_api_key)
    
    def _initialize_rerankers(self, cohere_api_key: Optional[str]):
        """Initialize reranking models with error handling."""
        # Cross encoder
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Loaded cross-encoder reranker")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")
        
        # BGE reranker
        try:
            self.bge_reranker = CrossEncoder('BAAI/bge-reranker-base')
            logger.info("Loaded BGE reranker")
        except Exception as e:
            logger.warning(f"Failed to load BGE reranker: {e}")
        
        # Cohere client
        if cohere_api_key and cohere:
            try:
                self.cohere_client = cohere.Client(cohere_api_key)
                logger.info("Initialized Cohere client")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere: {e}")
        
        # Sentence transformer for clustering
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer for clustering")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
    
    def cross_encoder_rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank using cross-encoder."""
        if not docs or not self.cross_encoder:
            return docs[:top_k]
        
        try:
            pairs = [(query, doc.page_content) for doc in docs]
            scores = self.cross_encoder.predict(pairs)
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:top_k]]
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")
            return docs[:top_k]
    
    def cohere_rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank using Cohere API."""
        if not self.cohere_client or not docs:
            return self.cross_encoder_rerank(query, docs, top_k)
        
        try:
            doc_texts = [doc.page_content for doc in docs]
            response = self.cohere_client.rerank(
                model='rerank-english-v2.0',
                query=query,
                documents=doc_texts,
                top_k=top_k
            )
            return [docs[r.index] for r in response.results]
        except Exception as e:
            logger.warning(f"Cohere reranking failed: {e}")
            return self.cross_encoder_rerank(query, docs, top_k)
    
    def bge_rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank using BGE reranker."""
        if not docs or not self.bge_reranker:
            return self.cross_encoder_rerank(query, docs, top_k)
        
        try:
            pairs = [(query, doc.page_content) for doc in docs]
            scores = self.bge_reranker.predict(pairs)
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:top_k]]
        except Exception as e:
            logger.warning(f"BGE reranking failed: {e}")
            return self.cross_encoder_rerank(query, docs, top_k)
    
    def clustering_filter(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """Select diverse documents using clustering."""
        if len(docs) <= top_k or not self.sentence_transformer:
            return docs[:top_k]
        
        try:
            embeddings = self.sentence_transformer.encode([doc.page_content for doc in docs])
            query_embedding = self.sentence_transformer.encode([query])
            
            n_clusters = min(top_k, len(docs))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            selected_docs = []
            for cluster_id in range(n_clusters):
                cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                if cluster_indices:
                    cluster_embeddings = embeddings[cluster_indices]
                    similarities = cosine_similarity(query_embedding, cluster_embeddings)[0]
                    best_idx = cluster_indices[np.argmax(similarities)]
                    selected_docs.append(docs[best_idx])
            
            return selected_docs
        except Exception as e:
            logger.warning(f"Clustering filter failed: {e}")
            return docs[:top_k]
    
    def redundancy_filter(self, query: str, docs: List[Document], top_k: int = 5, threshold: float = 0.8) -> List[Document]:
        """Filter out redundant documents."""
        if not docs or not self.sentence_transformer:
            return docs[:top_k]
        
        try:
            selected_docs = [docs[0]]
            selected_embeddings = [self.sentence_transformer.encode([docs[0].page_content])[0]]
            
            for doc in docs[1:]:
                if len(selected_docs) >= top_k:
                    break
                
                doc_embedding = self.sentence_transformer.encode([doc.page_content])[0]
                max_similarity = max(
                    cosine_similarity([doc_embedding], [emb])[0][0] 
                    for emb in selected_embeddings
                )
                
                if max_similarity < threshold:
                    selected_docs.append(doc)
                    selected_embeddings.append(doc_embedding)
            
            return selected_docs
        except Exception as e:
            logger.warning(f"Redundancy filter failed: {e}")
            return docs[:top_k]

class AdaptiveRAGSystem:
    """Main RAG system coordinating all components."""
    
    def __init__(self, gemini_api_key: str, cohere_api_key: Optional[str] = None):
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        self.query_analyzer = QueryAnalyzer(gemini_api_key)
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        
        self.documents = []
        self.retriever = None
        self.reranker = AdvancedReranker(cohere_api_key)
    
    def load_documents(self, file_path: str, file_type: str) -> None:
        """Load and process documents."""
        try:
            self.documents = self.doc_processor.load_document(file_path, file_type)
            self.retriever = AdvancedRetriever(self.documents, self.embedding_manager)
            logger.info(f"Successfully loaded {len(self.documents)} document chunks")
        except Exception as e:
            logger.error(f"Document loading failed: {e}")
            raise
    
    def retrieve_documents(self, query: str, config: RAGConfig) -> List[Document]:
        """Retrieve documents using specified strategy."""
        if not self.retriever:
            raise ValueError("No documents loaded")
        
        strategy_methods = {
            RetrievalStrategy.SEMANTIC_SIMILARITY: self.retriever.semantic_similarity_retrieval,
            RetrievalStrategy.BM25_KEYWORD: self.retriever.bm25_retrieval,
            RetrievalStrategy.HYBRID_ENSEMBLE: self.retriever.hybrid_ensemble_retrieval,
            RetrievalStrategy.RAG_FUSION: lambda q, k: self.retriever.rag_fusion_retrieval(q, k, self.gemini_model),
            RetrievalStrategy.HYDE: lambda q, k: self.retriever.hyde_retrieval(q, k, self.gemini_model),
        }
        
        method = strategy_methods.get(
            config.retrieval_strategy, 
            self.retriever.semantic_similarity_retrieval
        )
        return method(query, config.top_k)
    
    def rerank_documents(self, query: str, docs: List[Document], config: RAGConfig) -> List[Document]:
        """Rerank documents using specified method."""
        if not docs or not self.reranker:
            return docs[:config.rerank_top_k]
        
        rerank_methods = {
            RerankingMethod.CROSS_ENCODER: self.reranker.cross_encoder_rerank,
            RerankingMethod.COHERE_RERANK: self.reranker.cohere_rerank,
            RerankingMethod.BGE_RERANK: self.reranker.bge_rerank,
            RerankingMethod.CLUSTERING_FILTER: self.reranker.clustering_filter,
            RerankingMethod.REDUNDANCY_FILTER: self.reranker.redundancy_filter,
        }
        
        method = rerank_methods.get(
            config.reranking_method, 
            self.reranker.cross_encoder_rerank
        )
        return method(query, docs, config.rerank_top_k)
    
    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generate answer using Gemini with retrieved context."""
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
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process query through full RAG pipeline."""
        if not self.documents:
            return {"error": "No documents loaded"}
        
        try:
            # Analyze query and determine strategy
            config = self.query_analyzer.determine_strategy(question, len(self.documents))
            query_type = self.query_analyzer.categorize_query(question)
            
            # Retrieve documents
            retrieved_docs = self.retrieve_documents(question, config)
            
            # Rerank documents
            final_docs = self.rerank_documents(question, retrieved_docs, config)
            
            # Generate answer
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

# ==== Streamlit Application ====

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Advanced Adaptive RAG System",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Advanced Adaptive RAG System")
    st.markdown("""
    Upload a PDF or Word document and ask questions. The system automatically selects 
    the best retrieval and reranking strategy based on your query type.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        gemini_api_key = st.text_input(
            "Gemini API Key", 
            type="password",
            help="Required for document analysis and answer generation"
        )
        
        cohere_api_key = st.text_input(
            "Cohere API Key (Optional)", 
            type="password",
            help="Optional: Enables advanced reranking with Cohere"
        )
        
        if not gemini_api_key:
            st.warning("⚠️ Please enter your Gemini API key to continue.")
            st.info("Get your API key from Google AI Studio")
            return
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        try:
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = AdaptiveRAGSystem(gemini_api_key, cohere_api_key)
            st.success("✅ RAG system initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            return
    
    # Document upload section
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=['pdf', 'docx', 'doc'],
        help="Upload a PDF or Word document to query against"
    )
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                st.session_state.rag_system.load_documents(tmp_file_path, file_ext)
                
                num_chunks = len(st.session_state.rag_system.documents)
                st.success(f"✅ Successfully processed document into {num_chunks} chunks!")
                
                # Show document stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Document Chunks", num_chunks)
                with col2:
                    avg_length = np.mean([len(doc.page_content) for doc in st.session_state.rag_system.documents])
                    st.metric("Avg Chunk Length", f"{avg_length:.0f} chars")
                with col3:
                    total_length = sum(len(doc.page_content) for doc in st.session_state.rag_system.documents)
                    st.metric("Total Content", f"{total_length:,} chars")
                
            except Exception as e:
                st.error(f"❌ Error processing document: {e}")
            finally:
                os.unlink(tmp_file_path)
    
    # Query interface
    if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system.documents:
        st.header("❓ Ask Questions")
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Ask any question about your uploaded document..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Get Answer", type="primary")
        with col2:
            if st.button("🧹 Clear Results"):
                if 'query_results' in st.session_state:
                    del st.session_state.query_results
        
        # Process query
        if ask_button and query.strip():
            with st.spinner("Processing your question..."):
                result = st.session_state.rag_system.query(query)
                st.session_state.query_results = result
        
        # Display results
        if 'query_results' in st.session_state:
            result = st.session_state.query_results
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                # Answer section
                st.subheader("📝 Answer")
                st.write(result["answer"])
                
                # Metrics section
                st.subheader("📊 Processing Details")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Query Type", result["query_type"].replace('_', ' ').title())
                with col2:
                    st.metric("Retrieval Strategy", result["retrieval_strategy"].replace('_', ' ').title())
                with col3:
                    st.metric("Reranking Method", result["reranking_method"].replace('_', ' ').title())
                
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Documents Retrieved", result["num_retrieved"])
                with col5:
                    st.metric("Final Documents Used", result["num_final"])
                
                # Retrieved documents section
                if st.checkbox("📄 Show Retrieved Documents", key="show_docs"):
                    st.subheader("Retrieved Document Chunks")
                    
                    for i, doc in enumerate(result["retrieved_documents"]):
                        with st.expander(f"📄 Document Chunk {i+1}", expanded=False):
                            st.write(doc["content"])
                            
                            if doc["metadata"]:
                                st.subheader("Metadata")
                                for key, value in doc["metadata"].items():
                                    st.write(f"**{key}:** {value}")
        
        elif ask_button and not query.strip():
            st.warning("⚠️ Please enter a question to get started.")
    
    else:
        st.info("👆 Please upload a document to start asking questions!")
    
    # Information section
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        ### 🎯 Adaptive RAG Features
        
        This system automatically selects the best retrieval and reranking strategy based on your question type:
        
        **Query Types:**
        - **Factual:** Direct questions seeking specific information
        - **Analytical:** Questions requiring explanation or analysis  
        - **Comparative:** Questions comparing different concepts
        - **Temporal:** Time-based or chronological questions
        - **Complex Reasoning:** Multi-step reasoning questions
        - **Summarization:** Requests for summaries or overviews
        - **Specific Entity:** Questions about particular entities
        
        **Retrieval Strategies:**
        - **Semantic Similarity:** Uses embeddings to find contextually similar content
        - **BM25 Keyword:** Traditional keyword-based search (good for factual queries)
        - **Hybrid Ensemble:** Combines semantic and keyword approaches
        - **RAG Fusion:** Generates multiple query variations for comprehensive retrieval
        - **HyDE:** Creates hypothetical answers to improve retrieval accuracy
        
        **Reranking Methods:**
        - **Cross Encoder:** Neural model for query-document relevance scoring
        - **Cohere Rerank:** Advanced commercial reranking API
        - **BGE Rerank:** Open-source reranking model
        - **Clustering Filter:** Ensures diversity in selected documents
        - **Redundancy Filter:** Removes highly similar content
        
        ### 🔧 Technical Details
        
        - **Chunking:** Documents are split into overlapping chunks for better retrieval
        - **Embeddings:** Multiple embedding models with automatic fallback
        - **Error Handling:** Robust error handling with graceful degradation
        - **Caching:** Efficient caching of embeddings and models
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, LangChain, and Google Gemini")

if __name__ == "__main__":
    main()