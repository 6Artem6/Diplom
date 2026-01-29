-- ============================================
-- RAG-SPECIFIC TABLES
-- ============================================

-- 1. Embeddings и векторные представления
CREATE TABLE entity_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL,
    entity_type VARCHAR(100) NOT NULL, -- BusinessEntity, ProcessPattern, Screenshot, etc.
    embedding_model VARCHAR(100) NOT NULL, -- text-embedding-3-large, nomic-embed-text
    embedding_vector VECTOR(1536), -- pgvector extension
    embedding_metadata JSONB, -- {context: "checkout_flow", modality: "text"}
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(entity_id, entity_type, embedding_model)
);

CREATE INDEX idx_entity_embeddings_vector ON entity_embeddings
USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_entity_embeddings_type ON entity_embeddings(entity_type);

-- 2. Контекстные чанки для RAG
CREATE TABLE knowledge_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type VARCHAR(50) NOT NULL, -- business_entity, process_template, execution_log
    source_id UUID NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_metadata JSONB NOT NULL, -- {position: 0, parent_context: {...}}
    embedding_id UUID REFERENCES entity_embeddings(id),
    relevance_score FLOAT, -- для ранжирования
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_knowledge_chunks_source ON knowledge_chunks(source_type, source_id);
CREATE INDEX idx_knowledge_chunks_relevance ON knowledge_chunks(relevance_score DESC);

-- 3. RAG retrieval история и метрики
CREATE TABLE rag_retrieval_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    query_embedding_id UUID REFERENCES entity_embeddings(id),
    retrieval_strategy VARCHAR(50) NOT NULL, -- semantic, hybrid, rerank
    retrieved_chunk_ids UUID[] NOT NULL,
    relevance_scores FLOAT[] NOT NULL,
    rerank_model VARCHAR(100), -- cohere-rerank-v3, cross-encoder
    final_context_token_count INTEGER,
    retrieval_time_ms INTEGER,
    user_feedback VARCHAR(20), -- relevant, partially_relevant, irrelevant
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_rag_logs_created_at ON rag_retrieval_logs(created_at DESC);
CREATE INDEX idx_rag_logs_strategy ON rag_retrieval_logs(retrieval_strategy);

-- ============================================
-- MCP (Model Context Protocol) TABLES
-- ============================================

-- 4. MCP серверы и ресурсы
CREATE TABLE mcp_servers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    server_name VARCHAR(100) NOT NULL UNIQUE,
    server_type VARCHAR(50) NOT NULL, -- filesystem, database, api, custom
    connection_config JSONB NOT NULL, -- {url: "...", auth: {...}}
    capabilities JSONB NOT NULL, -- {tools: [...], resources: [...], prompts: [...]}
    health_check_endpoint VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    last_health_check TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 5. MCP ресурсы (файлы, данные, контекст)
CREATE TABLE mcp_resources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mcp_server_id UUID REFERENCES mcp_servers(id),
    resource_uri VARCHAR(500) NOT NULL UNIQUE, -- file:///path, db://table, api://endpoint
    resource_type VARCHAR(50) NOT NULL, -- document, image, structured_data
    resource_metadata JSONB, -- {mime_type: "...", size: 1024, tags: [...]}
    content_hash VARCHAR(64), -- для cache invalidation
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_mcp_resources_server ON mcp_resources(mcp_server_id);
CREATE INDEX idx_mcp_resources_type ON mcp_resources(resource_type);
CREATE INDEX idx_mcp_resources_uri ON mcp_resources(resource_uri);

-- 6. MCP инструменты (tools)
CREATE TABLE mcp_tools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mcp_server_id UUID REFERENCES mcp_servers(id),
    tool_name VARCHAR(100) NOT NULL,
    tool_description TEXT NOT NULL,
    input_schema JSONB NOT NULL, -- JSON Schema для параметров
    output_schema JSONB,
    execution_context JSONB, -- {requires_auth: true, rate_limit: {...}}
    usage_count INTEGER DEFAULT 0,
    avg_execution_time_ms INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(mcp_server_id, tool_name)
);

-- 7. MCP tool executions лог
CREATE TABLE mcp_tool_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mcp_tool_id UUID REFERENCES mcp_tools(id),
    execution_id VARCHAR(100) NOT NULL,
    input_parameters JSONB NOT NULL,
    execution_result JSONB,
    status VARCHAR(20) NOT NULL, -- success, failure, timeout
    execution_time_ms INTEGER,
    error_details JSONB,
    triggered_by VARCHAR(100), -- llm_agent, user, scheduled_job
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_mcp_executions_tool ON mcp_tool_executions(mcp_tool_id);
CREATE INDEX idx_mcp_executions_status ON mcp_tool_executions(status);
CREATE INDEX idx_mcp_executions_created_at ON mcp_tool_executions(created_at DESC);

-- ============================================
-- MULTIMODAL SCREENSHOT ANALYSIS
-- ============================================

-- 8. Screenshot хранилище и метаданные
CREATE TABLE screenshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    screenshot_hash VARCHAR(64) NOT NULL UNIQUE,
    storage_url VARCHAR(500) NOT NULL, -- S3/local path
    capture_timestamp TIMESTAMP NOT NULL,
    page_url TEXT,
    page_title TEXT,
    viewport_size JSONB, -- {width: 1920, height: 1080}
    device_type VARCHAR(50), -- desktop, mobile, tablet
    browser_context JSONB, -- {user_agent: "...", cookies: [...]}
    raw_metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_screenshots_hash ON screenshots(screenshot_hash);
CREATE INDEX idx_screenshots_capture_time ON screenshots(capture_timestamp DESC);

-- 9. Мультимодальный анализ скриншотов (GPT-4V/Claude Vision)
CREATE TABLE screenshot_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    screenshot_id UUID REFERENCES screenshots(id),
    analysis_model VARCHAR(100) NOT NULL, -- gpt-4-vision, claude-3-opus
    analysis_type VARCHAR(50) NOT NULL, -- element_detection, layout_analysis, text_extraction
    analysis_result JSONB NOT NULL,
    confidence_score FLOAT,
    processing_time_ms INTEGER,
    cost_usd DECIMAL(10,6),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 10. Обнаруженные UI элементы
CREATE TABLE detected_ui_elements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    screenshot_analysis_id UUID REFERENCES screenshot_analysis(id),
    element_type VARCHAR(50) NOT NULL, -- button, input, text, image, dropdown
    bounding_box JSONB NOT NULL, -- {x: 100, y: 200, width: 150, height: 40}
    text_content TEXT,
    visual_features JSONB, -- {color: "...", font_size: 14, has_icon: true}
    accessibility_attributes JSONB, -- {aria_label: "...", role: "..."}
    confidence FLOAT NOT NULL,
    element_hierarchy JSONB, -- {parent: "...", children: [...]}
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_ui_elements_screenshot_analysis ON detected_ui_elements(screenshot_analysis_id);
CREATE INDEX idx_ui_elements_type ON detected_ui_elements(element_type);

-- 11. Связывание UI элементов с бизнес-сущностями (cross-view linking)
CREATE TABLE ui_to_entity_mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detected_ui_element_id UUID REFERENCES detected_ui_elements(id),
    cross_view_entity_id UUID REFERENCES cross_view_entities(id),
    linking_method VARCHAR(50) NOT NULL, -- visual_similarity, text_matching, spatial_context
    confidence_score FLOAT NOT NULL,
    evidence JSONB NOT NULL, -- {visual_features: {...}, text_similarity: 0.9}
    validation_status VARCHAR(20), -- auto_confirmed, human_verified, rejected
    validated_by VARCHAR(100),
    validated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_ui_entity_mappings_element ON ui_to_entity_mappings(detected_ui_element_id);
CREATE INDEX idx_ui_entity_mappings_entity ON ui_to_entity_mappings(cross_view_entity_id);
CREATE INDEX idx_ui_entity_mappings_confidence ON ui_to_entity_mappings(confidence_score DESC);

-- ============================================
-- PLATFORM ADAPTERS
-- ============================================

-- 12. Платформенные адаптеры (Playwright, Appium, etc.)
CREATE TABLE platform_adapters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    adapter_name VARCHAR(100) NOT NULL UNIQUE,
    platform_type VARCHAR(50) NOT NULL, -- web, desktop, mobile, api
    automation_framework VARCHAR(50) NOT NULL, -- playwright, selenium, appium, pyautogui
    capabilities JSONB NOT NULL, -- {supports_screenshots: true, supports_accessibility_tree: false}
    configuration JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 13. Selector стратегии для разных платформ
CREATE TABLE selector_strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform_adapter_id UUID REFERENCES platform_adapters(id),
    strategy_name VARCHAR(100) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL, -- css, xpath, accessibility_id, image_template
    priority INTEGER DEFAULT 100,
    fallback_chain JSONB, -- [{strategy: "css", selector: "..."}, {...}]
    success_rate FLOAT,
    avg_execution_time_ms INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 14. Выполнение действий через адаптеры
CREATE TABLE adapter_action_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform_adapter_id UUID REFERENCES platform_adapters(id),
    execution_id VARCHAR(100) NOT NULL,
    action_type VARCHAR(50) NOT NULL, -- click, type, scroll, screenshot
    target_selector JSONB NOT NULL, -- {strategy: "css", value: "#button"}
    action_parameters JSONB,
    execution_result JSONB,
    status VARCHAR(20) NOT NULL,
    retry_count INTEGER DEFAULT 0,
    execution_time_ms INTEGER,
    screenshot_before_id UUID REFERENCES screenshots(id),
    screenshot_after_id UUID REFERENCES screenshots(id),
    error_details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_adapter_executions_platform ON adapter_action_executions(platform_adapter_id);
CREATE INDEX idx_adapter_executions_status ON adapter_action_executions(status);
CREATE INDEX idx_adapter_executions_created_at ON adapter_action_executions(created_at DESC);

-- ============================================
-- PATTERN EXTRACTION FROM SCREENSHOTS
-- ============================================

-- 15. Визуальные паттерны, извлеченные из скриншотов
CREATE TABLE visual_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_name VARCHAR(200) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL, -- navigation_flow, form_filling, data_display
    source_screenshots UUID[] NOT NULL, -- array of screenshot IDs
    visual_signature JSONB NOT NULL, -- {layout_structure: {...}, color_scheme: {...}}
    recurring_elements JSONB, -- [{element_type: "button", frequency: 0.9}]
    spatial_relationships JSONB, -- граф пространственных отношений
    business_context JSONB,
    occurrence_count INTEGER DEFAULT 1,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 16. Связь визуальных паттернов с бизнес-процессами
CREATE TABLE visual_to_process_mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    visual_pattern_id UUID REFERENCES visual_patterns(id),
    process_template_id UUID REFERENCES process_templates(id),
    mapping_confidence FLOAT NOT NULL,
    validation_method VARCHAR(50), -- heuristic, ml_model, human_verified
    supporting_evidence JSONB,
    is_validated BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- PROMPT ENGINEERING & CACHING
-- ============================================

-- 17. Prompt templates для разных задач
CREATE TABLE prompt_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_name VARCHAR(200) NOT NULL UNIQUE,
    task_type VARCHAR(50) NOT NULL, -- intent_understanding, action_planning, validation
    template_content TEXT NOT NULL,
    variable_schema JSONB NOT NULL, -- {variables: [{name: "context", type: "object"}]}
    rag_context_slots JSONB, -- {slots: ["business_rules", "similar_patterns"]}
    model_specific_config JSONB, -- {gpt4: {...}, claude: {...}}
    success_metrics JSONB,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 18. Prompt cache для LLM responses
CREATE TABLE prompt_cache_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_hash VARCHAR(64) NOT NULL UNIQUE,
    prompt_template_id UUID REFERENCES prompt_templates(id),
    full_prompt_content TEXT NOT NULL,
    context_hash VARCHAR(64) NOT NULL, -- hash RAG context
    model_used VARCHAR(100) NOT NULL,
    response_content JSONB NOT NULL,
    token_count_input INTEGER,
    token_count_output INTEGER,
    cache_hit_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_prompt_cache_hash ON prompt_cache_entries(prompt_hash);
CREATE INDEX idx_prompt_cache_context ON prompt_cache_entries(context_hash);
CREATE INDEX idx_prompt_cache_expires ON prompt_cache_entries(expires_at);

-- ============================================
-- CONTEXT & SESSION MANAGEMENT
-- ============================================

-- 19. Multi-turn conversation context
CREATE TABLE conversation_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(100) NOT NULL,
    turn_number INTEGER NOT NULL,
    user_message TEXT NOT NULL,
    system_response JSONB NOT NULL,
    rag_retrieved_context JSONB, -- что было получено из RAG
    mcp_resources_used UUID[], -- использованные MCP ресурсы
    screenshot_ids UUID[], -- связанные скриншоты
    entities_mentioned UUID[], -- упомянутые бизнес-сущности
    execution_ids VARCHAR(100)[], -- triggered executions
    context_state JSONB NOT NULL, -- состояние на момент turn
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(session_id, turn_number)
);

CREATE INDEX idx_conversation_session ON conversation_contexts(session_id);
CREATE INDEX idx_conversation_created_at ON conversation_contexts(created_at DESC);

-- 20. Context compression metrics
CREATE TABLE context_compression_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_context_tokens INTEGER NOT NULL,
    compressed_context_tokens INTEGER NOT NULL,
    compression_ratio FLOAT NOT NULL,
    compression_method VARCHAR(50) NOT NULL, -- summarization, extraction, pruning
    information_loss_score FLOAT, -- оценка потери информации
    execution_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- ADDITIONAL INDEXES FOR PERFORMANCE
-- ============================================

CREATE INDEX idx_screenshot_analysis_model ON screenshot_analysis(analysis_model);
CREATE INDEX idx_detected_ui_elements_confidence ON detected_ui_elements(confidence DESC);
CREATE INDEX idx_visual_patterns_type ON visual_patterns(pattern_type);
CREATE INDEX idx_platform_adapters_type ON platform_adapters(platform_type);
CREATE INDEX idx_mcp_servers_active ON mcp_servers(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_mcp_tools_usage ON mcp_tools(usage_count DESC);

-- Materialized view для RAG performance
CREATE MATERIALIZED VIEW mv_rag_effectiveness AS
SELECT
    retrieval_strategy,
    DATE_TRUNC('day', created_at) as date,
    COUNT(*) as total_retrievals,
    AVG(retrieval_time_ms) as avg_retrieval_time,
    COUNT(*) FILTER (WHERE user_feedback = 'relevant') as relevant_count,
    COUNT(*) FILTER (WHERE user_feedback = 'relevant')::FLOAT / COUNT(*)::FLOAT as relevance_rate
FROM rag_retrieval_logs
WHERE user_feedback IS NOT NULL
GROUP BY retrieval_strategy, DATE_TRUNC('day', created_at);

CREATE UNIQUE INDEX ON mv_rag_effectiveness (retrieval_strategy, date);