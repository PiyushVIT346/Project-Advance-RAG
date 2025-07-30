from langchain_huggingface import HuggingFaceEmbeddings
from logging_config.logger import logger


class EmbeddingManager:
    def __init__(self):
        self.embeddings = {}
        self.current_model = None
        self._initialize_embeddings()

    def _initialize_embeddings(self):
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

    def get_embeddings(self, model_name=None):
        model = model_name or self.current_model
        return self.embeddings.get(model, list(self.embeddings.values())[0])
