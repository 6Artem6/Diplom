import dagre from 'dagre'
import type { Node, Edge } from '@xyflow/react'
import type { NormalizedBpg } from './bpgNormalizer'

export type GraphNodeData = {
  label: string
  nodeKind: 'entity' | 'view'
  confidence?: number
  attributes?: Record<string, unknown>
}

export type GraphEdgeData = {
  edgeKind: string
  confidence?: number
  metadata?: Record<string, unknown>
}

const NODE_COLORS: Record<string, string> = {
  entity: '#3b82f6',
  view: '#10b981',
}

export function nodeColor(kind: string): string {
  return NODE_COLORS[kind] ?? '#94a3b8'
}

/**
 * Entity-level graph only: Entity, View, aggregated cross-view relationships.
 * No manifestation-level nodes or raw cross_view edge flood.
 */
export function buildFlowGraphFromNormalized(model: NormalizedBpg): {
  nodes: Node<GraphNodeData>[]
  edges: Edge<GraphEdgeData>[]
} {
  const nodes: Node<GraphNodeData>[] = []
  const edges: Edge<GraphEdgeData>[] = []
  const nodeIds = new Set<string>()

  const addNode = (node: Node<GraphNodeData>) => {
    if (nodeIds.has(node.id)) return
    nodeIds.add(node.id)
    nodes.push(node)
  }

  for (const entity of model.entities) {
    addNode({
      id: entity.id,
      type: 'default',
      position: { x: 0, y: 0 },
      data: {
        label: entity.type,
        nodeKind: 'entity',
        confidence: entity.confidence,
        attributes: {
          roles: entity.roles,
          views: entity.metrics.viewsCount,
          manifestations: entity.metrics.representationsCount,
        },
      },
      style: { borderColor: nodeColor('entity'), borderWidth: 2 },
    })
  }

  for (const view of model.views) {
    const viewNodeId = `view:${view.screenshot}`
    addNode({
      id: viewNodeId,
      type: 'default',
      position: { x: 0, y: 0 },
      data: {
        label: view.screenshot,
        nodeKind: 'view',
        attributes: {
          elements: view.elements.length,
          bbox_elements: view.bboxElements.length,
          view_uuid: view.view_uuid,
        },
      },
      style: { borderColor: nodeColor('view'), borderWidth: 2 },
    })
  }

  for (const entity of model.entities) {
    const seenViews = new Set<string>()
    for (const m of entity.elements) {
      if (!m.screenshot_id || seenViews.has(m.screenshot_id)) continue
      seenViews.add(m.screenshot_id)
      const viewNodeId = `view:${m.screenshot_id}`
      if (nodeIds.has(viewNodeId)) {
        edges.push({
          id: `appears-${viewNodeId}-${entity.id}`,
          source: viewNodeId,
          target: entity.id,
          label: 'contains',
          data: { edgeKind: 'contains' },
        })
      }
    }
  }

  for (const link of model.entityPairLinks) {
    edges.push({
      id: `cv-${link.entity_a_id}-${link.entity_b_id}`,
      source: link.entity_a_id,
      target: link.entity_b_id,
      label: `cross_view (${link.count})`,
      data: {
        edgeKind: 'cross_view',
        confidence: link.avg_similarity,
        metadata: {
          count: link.count,
          min: link.min_similarity,
          max: link.max_similarity,
          avg: link.avg_similarity,
        },
      },
      animated: true,
      style: { stroke: '#f97316' },
    })
  }

  return layoutGraph(nodes, edges)
}

function layoutGraph(
  nodes: Node<GraphNodeData>[],
  edges: Edge<GraphEdgeData>[],
): { nodes: Node<GraphNodeData>[]; edges: Edge<GraphEdgeData>[] } {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'TB', nodesep: 80, ranksep: 100 })

  for (const node of nodes) {
    g.setNode(node.id, { width: 200, height: 60 })
  }
  for (const edge of edges) {
    if (g.hasNode(edge.source) && g.hasNode(edge.target)) {
      g.setEdge(edge.source, edge.target)
    }
  }

  dagre.layout(g)

  const laidOut = nodes.map((node) => {
    const pos = g.node(node.id)
    if (!pos) return node
    return {
      ...node,
      position: { x: pos.x - 100, y: pos.y - 30 },
    }
  })

  return { nodes: laidOut, edges }
}
