# Multimodal embeddings для UI элементов
multimodal_ui_vector = {
    "id": "ui_element_screenshot123_button_add_to_cart",
    "values": [...],  # Combined visual + textual embedding
    "metadata": {
        "element_type": "button",
        "screenshot_id": "screenshot_uuid_123",
        "bbox": {"x": 100, "y": 200, "width": 150, "height": 40},
        "text_content": "Add to Cart",
        "visual_features": {
            "color": "#007BFF",
            "font_size": 16,
            "has_icon": True,
            "shape": "rounded_rectangle",
        },
        "spatial_context": {
            "position": "center_bottom",
            "nearby_elements": ["product_image", "price_text"],
        },
        "business_entity_id": "entity_cart_action",
        "linking_confidence": 0.91,
        # For cross-view linking
        "view_variations": [
            {"page_type": "product_detail", "occurrence_rate": 0.95},
            {"page_type": "cart", "occurrence_rate": 0.1},
        ],
    },
}

# Process pattern embeddings с visual context
process_pattern_with_visual = {
    "id": "pattern_checkout_flow_visual_v3",
    "values": [...],
    "metadata": {
        "pattern_type": "transactional_flow",
        "domain": "e-commerce",
        "visual_signature": {
            "typical_layout": "vertical_form",
            "color_scheme": ["#FFFFFF", "#007BFF", "#28A745"],
            "key_visual_elements": ["credit_card_icons", "progress_bar"],
        },
        "screenshot_examples": ["screenshot_uuid_123", "screenshot_uuid_456"],
        "ui_elements_sequence": [
            {"element_type": "input", "label": "card_number"},
            {"element_type": "input", "label": "expiry_date"},
            {"element_type": "button", "label": "complete_purchase"},
        ],
        "platform_adaptations": {
            "web": {"success_rate": 0.94, "avg_time_s": 12},
            "mobile": {"success_rate": 0.87, "avg_time_s": 18},
        },
    },
}

# MCP resource embeddings
mcp_resource_vector = {
    "id": "mcp_resource_customer_database_customers_table",
    "values": [...],
    "metadata": {
        "mcp_server": "postgres_server_prod",
        "resource_uri": "db://customers/table/customers",
        "resource_type": "structured_data",
        "schema": {
            "columns": ["id", "name", "email", "tier"],
            "data_types": ["uuid", "text", "text", "text"],
        },
        "access_patterns": ["customer_lookup_by_email", "customer_tier_validation"],
        "relevance_keywords": ["customer", "user", "account", "profile"],
        "last_updated": "2024-01-01T00:00:00Z",
    },
}

# Hybrid search query для RAG
hybrid_rag_query = {
    "vector": [...],  # Query embedding
    "sparse_vector": {"indices": [12, 45, 78, 123], "values": [0.9, 0.7, 0.6, 0.4]},
    "filter": {
        "$and": [
            {"domain": {"$eq": "e-commerce"}},
            {"pattern_type": {"$in": ["transactional_flow", "navigation_flow"]}},
            {
                "$or": [
                    {"validation_status": {"$eq": "human_verified"}},
                    {"confidence": {"$gte": 0.85}},
                ]
            },
            {"platform_adaptations.web.success_rate": {"$gte": 0.8}},
        ]
    },
    "top_k": 10,
    "include_metadata": True,
}

# Visual similarity search
visual_similarity_query = {
    "vector": [...],  # Visual embedding from current screenshot
    "filter": {
        "element_type": {"$eq": "button"},
        "visual_features.color": {"$in": ["#007BFF", "#0056B3"]},
        "business_entity_id": {"$exists": True},
    },
    "top_k": 5,
    "include_metadata": True,
}
