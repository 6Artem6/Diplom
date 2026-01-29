from sentence_transformers import SentenceTransformer


class ElementSemanticEmbedding:
    def __init__(self, vector_db):
        self.encoder = SentenceTransformer("all-mpnet-base-v2")
        self.vector_db = vector_db

    def create_embedding(self, element, node_id):
        text_features = [
            getattr(element, "text_content", None),
            getattr(element, "placeholder", None),
            getattr(element, "label_text", None),
            getattr(element, "aria_label", None),
            " ".join(getattr(element, "css_classes", [])),
        ]
        context_text = " ".join(filter(None, text_features))
        embedding = self.encoder.encode(context_text)

        self.vector_db.add(
            id=node_id, vector=embedding, metadata={"text": context_text}
        )
        return embedding

    def batch_create_embeddings(self, items):
        for item in items:
            self.create_embedding(item["element"], item["node_id"])

    def find_similar_elements(self, element, threshold=0.8):
        embedding = self.create_embedding(element, "temp")
        return self.vector_db.similarity_search(embedding, threshold)
