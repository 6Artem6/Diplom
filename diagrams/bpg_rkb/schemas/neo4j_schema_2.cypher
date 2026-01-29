// ============================================
// MCP INTEGRATION GRAPH
// ============================================

// MCP Server nodes
(:MCPServer {
  server_id: "mcp_postgres_prod",
  server_type: "database",
  connection_uri: "postgresql://...",
  capabilities: ["resources", "tools"],
  health_status: "healthy",
  last_health_check: datetime("2024-01-01T00:00:00Z")
})

// MCP Resources as graph nodes
(:MCPResource {
  resource_id: "customers_table",
  resource_uri: "db://customers/table",
  resource_type: "structured_data",
  schema: {...},
  access_count: 1234
})

// MCP Tools as graph nodes
(:MCPTool {
  tool_id: "validate_customer_tier",
  tool_name: "validate_customer_tier",
  input_schema: {...},
  output_schema: {...},
  usage_count: 567
})

// Relationships
(:MCPServer)-[:PROVIDES_RESOURCE]->(:MCPResource)
(:MCPServer)-[:PROVIDES_TOOL]->(:MCPTool)
(:MCPTool)-[:ACCESSES_RESOURCE]->(:MCPResource)
(:BusinessEntity)-[:MAPPED_TO_MCP_RESOURCE {
  mapping_confidence: 0.92,
  sync_strategy: "real_time"
}]->(:MCPResource)

// ============================================
// SCREENSHOT & VISUAL ANALYSIS GRAPH
// ============================================

// Screenshot nodes
(:Screenshot {
  screenshot_id: "screenshot_123",
  capture_timestamp: datetime("2024-01-01T10:30:00Z"),
  page_url: "https://example.com/product/123",
  viewport_size: {width: 1920, height: 1080},
  device_type: "desktop"
})

// Detected UI Element nodes
(:UIElement {
  element_id: "button_add_to_cart_123",
  element_type: "button",
  bbox: {x: 100, y: 200, width: 150, height: 40},
  text_content: "Add to Cart",
  visual_features: {
    color: "#007BFF",
    font_size: 16
  },
  confidence: 0.95
})

// Visual Pattern nodes
(:VisualPattern {
  pattern_id: "checkout_layout_pattern",
  pattern_type: "form_layout",
  visual_signature: {
    layout_structure: "vertical_form",
    color_scheme: ["#FFFFFF", "#007BFF"]
  },
  occurrence_count: 45,
  confidence: 0.88
})

// Relationships
(:Screenshot)-[:CONTAINS_ELEMENT {
  detection_confidence: 0.95,
  analysis_model: "gpt-4-vision"
}]->(:UIElement)

(:UIElement)-[:LINKED_TO_ENTITY {
  linking_method: "multimodal_analysis",
  confidence: 0.91,
  evidence: {
    visual_similarity: 0.88,
    text_matching: 0.95,
    spatial_context: 0.90
  }
}]->(:BusinessEntity)

(:Screenshot)-[:EXHIBITS_PATTERN {
  match_confidence: 0.87,
  matched_elements: ["button_123", "input_456"]
}]->(:VisualPattern)

(:VisualPattern)-[:REPRESENTS_PROCESS {
  mapping_confidence: 0.89,
  validation_method: "heuristic"
}]->(:ProcessPattern)

// Spatial relationships between UI elements
(:UIElement)-[:POSITIONED_ABOVE {
  distance_pixels: 50,
  alignment: "center"
}]->(:UIElement)

(:UIElement)-[:VISUALLY_SIMILAR_TO {
  similarity_score: 0.85,
  shared_features: ["color", "shape", "size"]
}]->(:UIElement)

// ============================================
// PLATFORM ADAPTER GRAPH (продолжение)
// ============================================

// Platform Adapter nodes
(:PlatformAdapter {
  adapter_id: "playwright_web_adapter",
  platform_type: "web",
  automation_framework: "playwright",
  capabilities: {
    supports_screenshots: true,
    supports_accessibility_tree: true,
    supports_network_interception: true
  },
  is_active: true
})

// Selector Strategy nodes
(:SelectorStrategy {
  strategy_id: "css_primary_web",
  strategy_type: "css",
  priority: 1,
  success_rate: 0.94,
  avg_execution_time_ms: 45
})

// Platform-specific Action Mapping
(:ActionMapping {
  action_id: "click_web_playwright",
  action_type: "click",
  platform_implementation: {
    method: "page.click",
    options: {timeout: 5000, force: false}
  }
})

// Relationships
(:PlatformAdapter)-[:USES_STRATEGY {
  fallback_order: 1,
  context_conditions: {"element_has_id": true}
}]->(:SelectorStrategy)

(:PlatformAdapter)-[:SUPPORTS_ACTION {
  reliability_score: 0.96
}]->(:ActionMapping)

(:UIElement)-[:EXECUTABLE_VIA {
  selector: "#add-to-cart-btn",
  selector_strategy: "css",
  platform_adapter: "playwright_web_adapter",
  last_successful_execution: datetime("2024-01-01T10:30:00Z")
}]->(:PlatformAdapter)

(:ProcessPattern)-[:REQUIRES_ADAPTER {
  platform_type: "web",
  fallback_adapters: ["selenium_web_adapter"]
}]->(:PlatformAdapter)

// ============================================
// RAG KNOWLEDGE GRAPH INTEGRATION
// ============================================

// Knowledge Chunk nodes (для RAG retrieval)
(:KnowledgeChunk {
  chunk_id: "chunk_business_rule_cart_123",
  chunk_text: "Products must be in stock before adding to cart",
  source_type: "business_rule",
  token_count: 12,
  relevance_score: 0.92,
  embedding_id: "embedding_uuid_456"
})

// RAG Retrieval Context nodes
(:RAGContext {
  context_id: "rag_context_session_789",
  session_id: "session_789",
  retrieval_timestamp: datetime("2024-01-01T10:30:00Z"),
  total_chunks: 5,
  total_tokens: 1200,
  retrieval_strategy: "hybrid"
})

// Relationships
(:KnowledgeChunk)-[:EXTRACTED_FROM {
  extraction_method: "semantic_chunking",
  chunk_position: 0
}]->(:BusinessRule)

(:KnowledgeChunk)-[:EMBEDDED_WITH {
  embedding_model: "text-embedding-3-large",
  embedding_dimension: 1536
}]->(:EntityEmbedding)

(:RAGContext)-[:CONTAINS_CHUNK {
  relevance_rank: 1,
  similarity_score: 0.92
}]->(:KnowledgeChunk)

(:ProcessPattern)-[:ENRICHED_BY_RAG {
  retrieval_count: 45,
  avg_relevance: 0.87,
  improved_success_rate: 0.12
}]->(:RAGContext)

// Cross-referencing для semantic search
(:KnowledgeChunk)-[:SEMANTICALLY_SIMILAR_TO {
  cosine_similarity: 0.85,
  shared_concepts: ["inventory", "validation", "precondition"]
}]->(:KnowledgeChunk)

// ============================================
// PROMPT ENGINEERING GRAPH
// ============================================

// Prompt Template nodes
(:PromptTemplate {
  template_id: "intent_understanding_v3",
  task_type: "intent_understanding",
  template_content: "Given the user input...",
  rag_slots: ["business_rules", "similar_patterns"],
  model_compatibility: ["gpt-4", "claude-3.5"],
  success_rate: 0.91
})

// Prompt Assembly nodes (runtime)
(:PromptAssembly {
  assembly_id: "prompt_exec_123",
  execution_id: "exec_123",
  assembled_at: datetime("2024-01-01T10:30:00Z"),
  total_tokens: 2500,
  compression_applied: true,
  original_tokens: 3800
})

// Relationships
(:PromptTemplate)-[:REQUIRES_RAG_CONTEXT {
  slot_name: "business_rules",
  max_chunks: 3,
  max_tokens: 500
}]->(:RAGContext)

(:PromptTemplate)-[:INCORPORATES_MCP_DATA {
  resource_types: ["customer_data", "inventory_status"],
  max_resources: 2
}]->(:MCPResource)

(:PromptAssembly)-[:USES_TEMPLATE]->(:PromptTemplate)
(:PromptAssembly)-[:INCLUDES_RAG_CONTEXT]->(:RAGContext)
(:PromptAssembly)-[:REFERENCES_SCREENSHOT]->(:Screenshot)
(:PromptAssembly)-[:MENTIONS_ENTITY]->(:BusinessEntity)

// ============================================
// MULTI-TURN CONVERSATION GRAPH
// ============================================

// Conversation Turn nodes
(:ConversationTurn {
  turn_id: "turn_session789_3",
  session_id: "session_789",
  turn_number: 3,
  user_message: "Add product to cart",
  system_response: {...},
  timestamp: datetime("2024-01-01T10:30:00Z")
})

// Context Evolution tracking
(:ContextState {
  state_id: "context_turn_3",
  turn_number: 3,
  active_entities: ["product_123", "customer_456"],
  current_page: "product_detail",
  user_intent: "add_to_cart",
  confidence: 0.89
})

// Relationships
(:ConversationTurn)-[:HAS_CONTEXT_STATE]->(:ContextState)
(:ConversationTurn)-[:FOLLOWS_TURN {
  time_gap_seconds: 15
}]->(:ConversationTurn)

(:ConversationTurn)-[:TRIGGERED_EXECUTION]->(:ProcessExecution)
(:ConversationTurn)-[:RETRIEVED_FROM_RAG]->(:RAGContext)
(:ConversationTurn)-[:CAPTURED_SCREENSHOT]->(:Screenshot)
(:ConversationTurn)-[:REFERENCED_ENTITY]->(:BusinessEntity)

(:ContextState)-[:EVOLVED_FROM {
  changes: {
    "added_entities": ["product_123"],
    "updated_intent": "view_to_purchase"
  }
}]->(:ContextState)

// ============================================
// EXECUTION & ADAPTATION GRAPH
// ============================================

// Process Execution instance nodes
(:ProcessExecution {
  execution_id: "exec_123",
  process_pattern_id: "add_to_cart_flow",
  started_at: datetime("2024-01-01T10:30:00Z"),
  completed_at: datetime("2024-01-01T10:30:05Z"),
  status: "success",
  actual_steps: ["navigate", "click", "validate"],
  deviation_score: 0.05
})

// Execution Step nodes
(:ExecutionStep {
  step_id: "step_exec123_2",
  execution_id: "exec_123",
  step_order: 2,
  action_type: "click",
  target_element_id: "button_add_to_cart_123",
  status: "success",
  execution_time_ms: 250
})

// Adaptation Event nodes
(:AdaptationEvent {
  adaptation_id: "adapt_exec123_1",
  event_type: "ui_change_detected",
  trigger: "element_not_found",
  adaptation_strategy: "fallback_selector",
  success: true,
  timestamp: datetime("2024-01-01T10:30:02Z")
})

// Relationships
(:ProcessExecution)-[:INSTANTIATES_PATTERN]->(:ProcessPattern)
(:ProcessExecution)-[:EXECUTED_ON_PLATFORM]->(:PlatformAdapter)
(:ProcessExecution)-[:CONTAINS_STEP {
  step_order: 2
}]->(:ExecutionStep)

(:ExecutionStep)-[:TARGETS_UI_ELEMENT]->(:UIElement)
(:ExecutionStep)-[:VALIDATED_BY_RULE]->(:BusinessRule)
(:ExecutionStep)-[:CAPTURED_SCREENSHOT_BEFORE]->(:Screenshot)
(:ExecutionStep)-[:CAPTURED_SCREENSHOT_AFTER]->(:Screenshot)

(:ProcessExecution)-[:TRIGGERED_ADAPTATION]->(:AdaptationEvent)
(:AdaptationEvent)-[:USED_FALLBACK_STRATEGY]->(:SelectorStrategy)

// Execution deviation tracking
(:ProcessExecution)-[:DEVIATED_FROM_EXPECTED {
  deviation_type: "step_reordering",
  expected_path: ["step1", "step2", "step3"],
  actual_path: ["step1", "step3", "step2"],
  deviation_score: 0.15,
  impact: "minor"
}]->(:ProcessPattern)

// ============================================
// CROSS-VIEW ENTITY LINKING GRAPH
// ============================================

// Enhanced Cross-View Entity with multimodal evidence
(:CrossViewEntity)-[:OBSERVED_IN_VIEW {
  view_type: "product_detail_page",
  ui_element_id: "button_add_to_cart_123",
  visual_confidence: 0.91,
  textual_confidence: 0.95,
  spatial_confidence: 0.88,
  overall_confidence: 0.91
}]->(:UIElement)

(:CrossViewEntity)-[:SAME_ENTITY_AS {
  linking_method: "multimodal_analysis",
  evidence: {
    visual_similarity: 0.88,
    text_matching: 0.95,
    behavioral_consistency: 0.92
  },
  confidence: 0.92,
  validated_by: "ml_model_v3"
}]->(:CrossViewEntity)

// Visual manifestation tracking
(:CrossViewEntity)-[:APPEARS_VISUALLY_AS {
  page_context: "checkout",
  typical_position: "center_bottom",
  visual_variations: [
    {variation: "mobile", confidence: 0.87},
    {variation: "desktop", confidence: 0.94}
  ]
}]->(:VisualPattern)

// Business context enrichment
(:CrossViewEntity)-[:ENRICHED_BY_MCP {
  mcp_resource_uri: "db://customers/table/customers",
  data_fields: ["name", "email", "tier"],
  last_sync: datetime("2024-01-01T10:00:00Z"),
  sync_quality: 0.98
}]->(:MCPResource)

// ============================================
// LEARNING & FEEDBACK GRAPH
// ============================================

// User Feedback nodes
(:UserFeedback {
  feedback_id: "feedback_exec123",
  execution_id: "exec_123",
  feedback_type: "correction",
  sentiment_score: -0.3,
  user_comment: "Wrong element clicked",
  timestamp: datetime("2024-01-01T10:30:10Z")
})

// Learning Example nodes
(:LearningExample {
  example_id: "example_789",
  example_type: "entity_linking",
  input_data: {...},
  expected_output: {...},
  actual_output: {...},
  correctness_score: 0.75,
  difficulty_level: 3
})

// Model Performance Tracking
(:ModelPerformance {
  model_name: "gpt-4-vision",
  task_type: "ui_element_detection",
  measurement_period: "2024-01",
  accuracy: 0.93,
  avg_latency_ms: 850,
  cost_per_call_usd: 0.015
})

// Relationships
(:UserFeedback)-[:RELATES_TO_EXECUTION]->(:ProcessExecution)
(:UserFeedback)-[:CORRECTS_ENTITY_LINKING]->(:CrossViewEntity)
(:UserFeedback)-[:IMPROVES_PATTERN]->(:ProcessPattern)

(:LearningExample)-[:GENERATED_FROM_EXECUTION]->(:ProcessExecution)
(:LearningExample)-[:USED_IN_TRAINING {
  training_iteration: 5,
  contributed_to_improvement: 0.03
}]->(:ModelPerformance)

(:ProcessPattern)-[:PERFORMANCE_TRACKED_BY]->(:ModelPerformance)
(:UIElement)-[:DETECTION_QUALITY_BY]->(:ModelPerformance)

// ============================================
// COMPLEX QUERIES FOR RAG/MCP/MULTIMODAL
// ============================================

// Cypher query для RAG-enhanced pattern retrieval
MATCH (intent:UserIntent {intent_type: "add_to_cart"})
-[:REQUIRES_PATTERN]->(pattern:ProcessPattern)
-[:ENRICHED_BY_RAG]->(rag:RAGContext)
-[:CONTAINS_CHUNK]->(chunk:KnowledgeChunk)
-[:EXTRACTED_FROM]->(rule:BusinessRule)
WHERE pattern.domain = "e-commerce"
  AND pattern.success_rate > 0.85
  AND rag.retrieval_strategy = "hybrid"
RETURN pattern,
       collect(DISTINCT chunk.chunk_text) as relevant_rules,
       avg(chunk.relevance_score) as avg_relevance

// Query для cross-view entity linking с multimodal evidence
MATCH (entity:CrossViewEntity)
-[link:OBSERVED_IN_VIEW]->(ui:UIElement)
<-[:CONTAINS_ELEMENT]-(screenshot:Screenshot)
WHERE link.overall_confidence > 0.85
  AND link.visual_confidence > 0.80
  AND link.textual_confidence > 0.90
OPTIONAL MATCH (entity)-[:ENRICHED_BY_MCP]->(mcp:MCPResource)
RETURN entity,
       collect({
         screenshot: screenshot.screenshot_id,
         ui_element: ui.element_id,
         confidence: link.overall_confidence,
         evidence: link.evidence
       }) as observations,
       mcp.resource_uri as external_data_source

// Query для platform adapter selection с context
MATCH (pattern:ProcessPattern {pattern_id: "checkout_flow"})
-[:REQUIRES_ADAPTER]->(adapter:PlatformAdapter)
-[:USES_STRATEGY]->(strategy:SelectorStrategy)
WHERE adapter.is_active = true
  AND strategy.success_rate > 0.90
OPTIONAL MATCH (adapter)-[:SUPPORTS_ACTION]->(action:ActionMapping)
WITH adapter,
     collect(DISTINCT strategy) as strategies,
     collect(DISTINCT action) as actions
ORDER BY adapter.capabilities.supports_screenshots DESC,
         avg([s IN strategies | s.success_rate]) DESC
RETURN adapter, strategies, actions
LIMIT 3

// Query для prompt assembly с RAG + MCP context
MATCH (template:PromptTemplate {task_type: "action_planning"})
-[:REQUIRES_RAG_CONTEXT]->(rag_spec)
MATCH (current_context:ContextState {session_id: $session_id})
-[:ACTIVE_ENTITY]->(entity:BusinessEntity)
MATCH (entity)-[:LINKED_TO_ENTITY]->(ui:UIElement)
<-[:CONTAINS_ELEMENT]-(screenshot:Screenshot)
OPTIONAL MATCH (entity)-[:ENRICHED_BY_MCP]->(mcp:MCPResource)
WITH template,
     collect(DISTINCT entity) as entities,
     collect(DISTINCT ui) as ui_elements,
     collect(DISTINCT screenshot)[0..3] as recent_screenshots,
     collect(DISTINCT mcp) as mcp_resources
// Retrieve RAG context
CALL {
  WITH entities
  UNWIND entities as e
  MATCH (e)-[:GOVERNED_BY]->(rule:BusinessRule)
  -[:EXTRACTED_FROM]->(chunk:KnowledgeChunk)
  RETURN collect(chunk)[0..5] as relevant_chunks
}
RETURN template.template_content as base_template,
       entities,
       ui_elements,
       recent_screenshots,
       mcp_resources,
       relevant_chunks

// Query для learning loop: найти failed executions для улучшения
MATCH (exec:ProcessExecution {status: "failed"})
-[:INSTANTIATES_PATTERN]->(pattern:ProcessPattern)
-[:CONTAINS_STEP]->(step:ExecutionStep {status: "failed"})
-[:TARGETS_UI_ELEMENT]->(ui:UIElement)
OPTIONAL MATCH (exec)-[:TRIGGERED_ADAPTATION]->(adapt:AdaptationEvent)
OPTIONAL MATCH (exec)<-[:RELATES_TO_EXECUTION]-(feedback:UserFeedback)
WITH pattern, ui,
     count(exec) as failure_count,
     collect(DISTINCT step.action_type) as failed_actions,
     collect(DISTINCT adapt.adaptation_strategy) as tried_strategies,
     collect(DISTINCT feedback.user_comment) as user_comments
WHERE failure_count > 5
RETURN pattern.pattern_id,
       pattern.pattern_name,
       failure_count,
       failed_actions,
       tried_strategies,
       user_comments
ORDER BY failure_count DESC
LIMIT 10

// ============================================
// INDEXES FOR PERFORMANCE
// ============================================

CREATE INDEX screenshot_timestamp FOR (s:Screenshot) ON (s.capture_timestamp);
CREATE INDEX ui_element_type FOR (u:UIElement) ON (u.element_type);
CREATE INDEX ui_element_confidence FOR (u:UIElement) ON (u.confidence);
CREATE INDEX visual_pattern_type FOR (v:VisualPattern) ON (v.pattern_type);
CREATE INDEX mcp_server_health FOR (m:MCPServer) ON (m.health_status);
CREATE INDEX mcp_resource_uri FOR (m:MCPResource) ON (m.resource_uri);
CREATE INDEX adapter_platform FOR (a:PlatformAdapter) ON (a.platform_type);
CREATE INDEX selector_success FOR (s:SelectorStrategy) ON (s.success_rate);
CREATE INDEX knowledge_chunk_relevance FOR (k:KnowledgeChunk) ON (k.relevance_score);
CREATE INDEX execution_status FOR (e:ProcessExecution) ON (e.status);
CREATE INDEX conversation_session FOR (c:ConversationTurn) ON (c.session_id, c.turn_number);

CREATE FULLTEXT INDEX ui_element_text FOR (u:UIElement) ON EACH [u.text_content];
CREATE FULLTEXT INDEX knowledge_chunk_search FOR (k:KnowledgeChunk) ON EACH [k.chunk_text];