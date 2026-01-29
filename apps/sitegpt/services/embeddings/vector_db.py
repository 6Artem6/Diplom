from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams


class QdrantVectorDB:
    def __init__(self, host="localhost", port=6333, collection_name="elements"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collections_api.exists(self.collection_name):
            self.client.collections_api.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance="Cosine"),
            )

    def add(self, id, vector, metadata=None):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[{"id": id, "vector": vector.tolist(), "payload": metadata or {}}],
        )

    def similarity_search(self, vector, threshold=0.8):
        results = self.client.search(
            collection_name=self.collection_name, query_vector=vector.tolist(), limit=10
        )
        # фильтруем по threshold
        return [r for r in results if r.score >= threshold]
