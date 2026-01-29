-- ============================================
-- VECTOR SIMILARITY CACHE (для ускорения RAG)
-- ============================================

CREATE TABLE vector_similarity_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_vector_hash VARCHAR(64) NOT NULL,
    target_vector_hash VARCHAR(64) NOT NULL,
    similarity_score FLOAT NOT NULL,
    similarity_metric VARCHAR(20) NOT NULL, -- cosine, euclidean, dot_product
    computed_at TIMESTAMP DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    UNIQUE(query_vector_hash, target_vector_hash, similarity_metric)
);

CREATE INDEX idx_vector_sim_cache_query ON vector_similarity_cache(query_vector_hash);
CREATE INDEX idx_vector_sim_cache_score ON vector_similarity_cache(similarity_score DESC);

-- ============================================
-- RERANKER MODELS & RESULTS
-- ============================================

CREATE TABLE reranker_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL UNIQUE,
    model_type VARCHAR(50) NOT NULL, -- cross_encoder, listwise, pointwise
    api_endpoint VARCHAR(255),
    cost_per_1k_docs DECIMAL(10,6),
    avg_latency_ms INTEGER,
    accuracy_metrics JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE reranking_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rag_retrieval_log_id UUID REFERENCES rag_retrieval_logs(id),
    reranker_model_id UUID REFERENCES reranker_models(id),
    original_rankings JSONB NOT NULL, -- [{doc_id, original_score}, ...]
    reranked_results JSONB NOT NULL, -- [{doc_id, rerank_score}, ...]
    rank_changes JSONB, -- документация изменений
    improvement_score FLOAT,
    reranking_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- MULTIMODAL FUSION METADATA
-- ============================================

CREATE TABLE multimodal_fusion_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_name VARCHAR(100) NOT NULL UNIQUE,
    fusion_strategy VARCHAR(50) NOT NULL, -- early_fusion, late_fusion, hybrid
    modality_weights JSONB NOT NULL, -- {visual: 0.6, textual: 0.3, spatial: 0.1}
    fusion_model VARCHAR(100), -- model используемый для fusion
    performance_metrics JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE multimodal_fusion_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fusion_config_id UUID REFERENCES multimodal_fusion_configs(id),
    input_modalities JSONB NOT NULL, -- {visual: {...}, textual: {...}}
    fused_representation JSONB NOT NULL,
    confidence_by_modality JSONB,
    final_confidence FLOAT NOT NULL,
    fusion_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- DYNAMIC CONTEXT WINDOWS
-- ============================================

CREATE TABLE context_window_strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name VARCHAR(100) NOT NULL UNIQUE,
    window_size_tokens INTEGER NOT NULL,
    compression_method VARCHAR(50), -- sliding_window, attention_based, summarization
    priority_rules JSONB NOT NULL, -- rules для приоритизации контекста
    performance_metrics JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE context_window_applications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_context_id UUID REFERENCES conversation_contexts(id),
    strategy_id UUID REFERENCES context_window_strategies(id),
    original_context_tokens INTEGER NOT NULL,
    compressed_context_tokens INTEGER NOT NULL,
    compression_ratio FLOAT NOT NULL,
    information_retained_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- VISUAL GROUNDING (связь текста с визуальными элементами)
-- ============================================

CREATE TABLE visual_grounding_annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    screenshot_id UUID REFERENCES screenshots(id),
    text_reference TEXT NOT NULL, -- "the blue button"
    grounded_element_id UUID REFERENCES detected_ui_elements(id),
    grounding_confidence FLOAT NOT NULL,
    grounding_method VARCHAR(50), -- attention_maps, region_proposals
    bounding_box_adjustment JSONB, -- если bbox скорректирован
    validation_status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_visual_grounding_screenshot ON visual_grounding_annotations(screenshot_id);
CREATE INDEX idx_visual_grounding_element ON visual_grounding_annotations(grounded_element_id);

-- ============================================
-- INCREMENTAL LEARNING & MODEL UPDATES
-- ============================================

CREATE TABLE model_update_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    update_type VARCHAR(50) NOT NULL, -- fine_tuning, prompt_optimization, embedding_refresh
    training_data_size INTEGER,
    performance_before JSONB NOT NULL,
    performance_after JSONB NOT NULL,
    improvement_metrics JSONB,
    deployment_status VARCHAR(20), -- staged, production, rolled_back
    deployed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- COST TRACKING & OPTIMIZATION
-- ============================================

CREATE TABLE cost_allocation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID,
    cost_category VARCHAR(50) NOT NULL, -- llm_api, vector_db, storage, compute
    service_provider VARCHAR(100), -- openai, anthropic, pinecone
    cost_usd DECIMAL(10,6) NOT NULL,
    usage_units INTEGER, -- tokens, queries, GB, etc.
    unit_type VARCHAR(50),
    time_period_start TIMESTAMP NOT NULL,
    time_period_end TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_cost_allocation_tenant ON cost_allocation(tenant_id);
CREATE INDEX idx_cost_allocation_period ON cost_allocation(time_period_start, time_period_end);
CREATE INDEX idx_cost_allocation_category ON cost_allocation(cost_category);

-- Материализованное представление для cost analytics
CREATE MATERIALIZED VIEW mv_cost_by_tenant_daily AS
SELECT 
    tenant_id,
    DATE_TRUNC('day', time_period_start) as day,
    cost_category,
    SUM(cost_usd) as total_cost_usd,
    SUM(usage_units) as total_units
FROM cost_allocation
GROUP BY tenant_id, DATE_TRUNC('day', time_period_start), cost_category;

CREATE UNIQUE INDEX ON mv_cost_by_tenant_daily (tenant_id, day, cost_category);