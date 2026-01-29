# Vector Index Structure
index_config = {
    "dimension": 1536,  # OpenAI text-embedding-3-large
    "metric": "cosine",
    "pods": 1,
    "replicas": 1,
    "pod_type": "p1.x1"
}

# Vector Record Structure
vector_record = {
    "id": "pattern_e-commerce_add_to_cart_v1",
    "values": [0.123, -0.456, 0.789, ...],  # 1536-dim embedding
    "metadata": {
        "pattern_type": "transactional_flow",
        "domain": "e-commerce",
        "entity_types": ["product", "cart", "customer"],
        "complexity_score": 0.7,
        "success_rate": 0.94,
        "avg_execution_time": 3.2,
        "last_updated": "2024-01-01T00:00:00Z",
        "version": "1.0",
        "tags": ["verified", "production_ready"],
        "business_context": {
            "required_permissions": ["authenticated_user"],
            "preconditions": ["product_available"],
            "postconditions": ["cart_updated", "inventory_decremented"]
        }
    }
}

# Cross-View Entity Vectors
cross_view_vector = {
    "id": "entity_customer_profile_view_1",
    "values": [0.234, -0.567, 0.890, ...],  # Composite embedding
    "metadata": {
        "entity_id": "customer_123",
        "entity_type": "Customer",
        "view_context": "profile_page",
        "gui_elements": {
            "text_content": "John Doe, Premium Member",
            "element_types": ["text", "badge", "avatar"],
            "layout_position": "header_right"
        },
        "linking_confidence": 0.89,
        "screenshot_id": "screenshot_456",
        "bbox": {"x": 100, "y": 50, "width": 200, "height": 80},
        "visual_features_hash": "abc123def456",
        "created_at": "2024-01-01T00:00:00Z"
    }
}

# Query Examples
similarity_query = {
    "vector": [0.123, -0.456, 0.789, ...],
    "filter": {
        "domain": {"$eq": "e-commerce"},
        "pattern_type": {"$eq": "transactional_flow"},
        "success_rate": {"$gte": 0.8}
    },
    "top_k": 5,
    "include_metadata": True
}
