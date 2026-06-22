export interface Confidence {
  score: number
  method: string
  metadata: Record<string, unknown>
}

export interface Provenance {
  evidence_sources: string[]
  inference_method: string
  timestamp: string
  metadata: Record<string, unknown>
}

export interface DetectedElement {
  element_id: string
  view_id: string
  screenshot_id: string
  bbox: Record<string, number>
  class: string
  role?: string | null
  text?: string | null
  cleaned_text?: string | null
  ocr_text?: string | null
  noisy_ocr?: boolean
  embedding_input?: string | null
  embedding_source?: string | null
  confidence: number
  pipeline_stage: string
  entity_instance_id?: string | null
  manifestation_id?: string | null
}

export interface GUIManifestation {
  id: string
  entity_instance_id: string
  view_id: string
  screenshot_id: string
  bounding_box: Record<string, number>
  layout_features?: Record<string, unknown>
  container_bbox?: Record<string, number> | null
}

export interface EntityType {
  id: string
  name: string
  description?: string | null
  confidence: Confidence
  provenance: Provenance
}

export interface EntityInstance {
  id: string
  entity_type_id: string
  attributes: Record<string, unknown>
  confidence: Confidence
  provenance: Provenance
}

export interface Action {
  id: string
  action_type: string
  trigger_element: Record<string, unknown>
  confidence: Confidence
  provenance: Provenance
}

export interface PatternNode {
  id: string
  name: string
  steps: string[]
  confidence: Confidence
  provenance: Provenance
}

export interface Rule {
  id: string
  rule_type: string
  condition: string
  scope?: string | null
  confidence: Confidence
  provenance: Provenance
}

export interface BpgEdge {
  id: string
  source_id: string
  target_id: string
  edge_type: string
  confidence: Confidence
  provenance: Provenance
  metadata: Record<string, unknown>
  similarity_score?: number
  relationship_type?: string
  action_id?: string
  role?: string
}

export interface CrossViewEdge extends BpgEdge {
  edge_type: 'cross_view'
  similarity_score?: number
}

export interface BusinessProcessGraph {
  id: string
  entity_types: EntityType[]
  entity_instances: EntityInstance[]
  actions: Action[]
  patterns: PatternNode[]
  rules: Rule[]
  edges: BpgEdge[]
  cross_view_edges: CrossViewEdge[]
  detected_elements: DetectedElement[]
  gui_manifestations: GUIManifestation[]
  created_at: string
}

export interface BuildBpgResponse {
  bpg_id: string
  entity_types_count: number
  entity_instances_count: number
  actions_count: number
  patterns_count: number
  rules_count: number
  edges_count: number
  cross_view_edges_count: number
  detected_elements_count: number
  detected_elements: DetectedElement[]
  message: string
}

export interface VisualizationCrossViewEdge {
  source_id: string
  target_id: string
  similarity_score?: number
  confidence: number
  source_view_id?: string
  target_view_id?: string
  validation?: string
}

export interface VisualizationEntityInstance {
  id: string
  attributes: Record<string, unknown>
  view_count: number
  is_cross_view: boolean
  color: string
}

export interface VisualizationSummary {
  total_cross_view_edges: number
  cross_view_entities: number
  validation_passed: boolean
}

export interface SimilarityStats {
  threshold: number
  max: number
  mean: number
  min: number
  count: number
}

export interface VisualizationFile {
  filename: string
  url: string
}

export interface VisualizationResponse {
  bpg_id: string
  cross_view_edges: VisualizationCrossViewEdge[]
  entity_instances: VisualizationEntityInstance[]
  summary: VisualizationSummary
  similarity_stats: SimilarityStats | Record<string, never>
  visualization_files: VisualizationFile[]
  detected_elements?: DetectedElement[]
}

export type BuildStatus =
  | 'idle'
  | 'building'
  | 'loading_graph'
  | 'loading_visualization'
  | 'done'
  | 'error'
