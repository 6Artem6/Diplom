// Business Entity Nodes
CREATE CONSTRAINT business_entity_id FOR (e:BusinessEntity) REQUIRE e.entity_id IS UNIQUE;

// Entity types and relationships
(:BusinessEntity {
  entity_id: "customer_123",
  entity_type: "Customer",
  domain: "e-commerce",
  attributes: {
    name: "John Doe",
    email: "john@example.com",
    tier: "premium"
  },
  gui_manifestations: [
    {
      page_type: "profile",
      selectors: [".customer-name", "#user-profile"],
      typical_position: "header"
    },
    {
      page_type: "checkout",
      selectors: [".billing-name", ".customer-info"],
      typical_position: "form"
    }
  ]
})

// Cross-View Entity Relationships
(:BusinessEntity)-[:SAME_AS {confidence: 0.95, linking_method: "multimodal_analysis"}]->(:BusinessEntity)
(:BusinessEntity)-[:RELATED_TO {relationship_type: "contains", strength: 0.8}]->(:BusinessEntity)
(:BusinessEntity)-[:TRANSFORMS_TO {process_name: "order_creation", probability: 0.9}]->(:BusinessEntity)

// Process Pattern Nodes
(:ProcessPattern {
  pattern_id: "add_to_cart_flow",
  pattern_type: "transactional",
  domain: "e-commerce",
  steps: [
    {
      step_order: 1,
      action_type: "navigate",
      target_entity: "product",
      validation_rules: ["product_available", "user_authenticated"]
    },
    {
      step_order: 2,
      action_type: "interact",
      target_element: "add_to_cart_button",
      expected_outcome: "cart_updated"
    }
  ],
  success_criteria: {
    completion_rate: 0.95,
    error_rate: 0.02,
    avg_execution_time: 3.5
  }
})

// Business Rule Nodes
(:BusinessRule {
  rule_id: "inventory_check",
  rule_type: "precondition",
  applicable_patterns: ["add_to_cart_flow", "purchase_flow"],
  rule_definition: {
    condition: "product.inventory > 0",
    error_message: "Product out of stock",
    severity: "blocking"
  }
})

// Relationships between patterns and rules
(:ProcessPattern)-[:REQUIRES {rule_type: "precondition"}]->(:BusinessRule)
(:ProcessPattern)-[:VALIDATES {rule_type: "postcondition"}]->(:BusinessRule)
(:BusinessEntity)-[:GOVERNED_BY]->(:BusinessRule)

// Navigation and Flow Relationships
(:ProcessPattern)-[:FOLLOWS {probability: 0.8, typical_user_path: true}]->(:ProcessPattern)
(:ProcessPattern)-[:BRANCHES_TO {condition: "user_authenticated", probability: 0.6}]->(:ProcessPattern)
