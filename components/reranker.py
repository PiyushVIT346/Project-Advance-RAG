import numpy as np
from logging_config.logger import logger

try:
    import cohere
except ImportError:
    cohere = None

from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedReranker:
    def __init__(self, cohere_api_key=None):
        self.cross_encoder = None
        self.bge_reranker = None
        self.cohere_client = None
        self.sentence_transformer = None
        self._initialize_rerankers(cohere_api_key)

    def _initialize_rerankers(self, cohere_api_key):
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Loaded cross-encoder reranker")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")
        try:
            self.bge_reranker = CrossEncoder('BAAI/bge-reranker-base')
            logger.info("Loaded BGE reranker")
        except Exception as e:
            logger.warning(f"Failed to load BGE reranker: {e}")
        if cohere_api_key and cohere:
            try:
                self.cohere_client = cohere.Client(cohere_api_key)
                logger.info("Initialized Cohere client")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere: {e}")
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer for clustering")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")

    def cross_encoder_rerank(self, query, docs, top_k=5):
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

    def cohere_rerank(self, query, docs, top_k=5):
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

    def bge_rerank(self, query, docs, top_k=5):
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

    def clustering_filter(self, query, docs, top_k=5):
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

    def redundancy_filter(self, query, docs, top_k=5, threshold=0.8):
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
