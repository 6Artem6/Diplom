-- Business Entities and Templates
CREATE TABLE business_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(100) NOT NULL,
    domain VARCHAR(100) NOT NULL,
    attributes JSONB,
    constraints JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE process_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_name VARCHAR(200) NOT NULL,
    domain VARCHAR(100) NOT NULL,
    template_definition JSONB NOT NULL,
    parameters_schema JSONB,
    success_criteria JSONB,
    performance_metrics JSONB,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE business_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(200) NOT NULL,
    rule_definition JSONB NOT NULL,
    applicable_entities TEXT[],
    priority INTEGER DEFAULT 100,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Cross-View Entity Linking
CREATE TABLE cross_view_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id VARCHAR(100) NOT NULL UNIQUE,
    entity_type VARCHAR(100),
    confidence FLOAT NOT NULL,
    representations JSONB NOT NULL,
    business_attributes JSONB,
    analysis_metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE ui_element_representations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cross_view_entity_id UUID REFERENCES cross_view_entities(id),
    screenshot_id VARCHAR(100) NOT NULL,
    bbox JSONB NOT NULL, -- {x, y, width, height}
    ui_type VARCHAR(50),
    text_content TEXT,
    visual_features JSONB,
    layout_context JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Function Registry and Tools
CREATE TABLE function_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    function_name VARCHAR(100) NOT NULL UNIQUE,
    function_type VARCHAR(50) NOT NULL, -- business_logic, template_execution, validation
    description TEXT,
    parameters_schema JSONB NOT NULL,
    return_schema JSONB,
    implementation_details JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE execution_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id VARCHAR(100) NOT NULL,
    function_name VARCHAR(100),
    input_parameters JSONB,
    execution_result JSONB,
    execution_time_ms INTEGER,
    status VARCHAR(20) NOT NULL, -- success, failure, timeout
    error_details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Performance and Analytics
CREATE TABLE llm_request_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(100) NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    prompt_hash VARCHAR(64),
    token_count_input INTEGER,
    token_count_output INTEGER,
    response_time_ms INTEGER,
    cost_usd DECIMAL(10,6),
    cache_hit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_business_entities_domain ON business_entities(domain);
CREATE INDEX idx_business_entities_type ON business_entities(entity_type);
CREATE INDEX idx_cross_view_entities_confidence ON cross_view_entities(confidence);
CREATE INDEX idx_execution_logs_status ON execution_logs(status);
CREATE INDEX idx_execution_logs_created_at ON execution_logs(created_at);
CREATE INDEX idx_llm_logs_model ON llm_request_logs(model_used);
CREATE INDEX idx_llm_logs_cache_hit ON llm_request_logs(cache_hit);