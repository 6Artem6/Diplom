import { useCallback, useMemo, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { NormalizedBpg } from '@/lib/bpgNormalizer'
import {
  buildFlowGraphFromNormalized,
  nodeColor,
  type GraphEdgeData,
  type GraphNodeData,
} from '@/lib/graphBuilder'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

export function GraphTab({ model }: { model: NormalizedBpg | null }) {
  const [selectedNode, setSelectedNode] = useState<Node<GraphNodeData> | null>(null)
  const [selectedEdge, setSelectedEdge] = useState<Edge<GraphEdgeData> | null>(null)

  const { nodes, edges } = useMemo(() => {
    if (!model) return { nodes: [], edges: [] }
    return buildFlowGraphFromNormalized(model)
  }, [model])

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node<GraphNodeData>) => {
    setSelectedEdge(null)
    setSelectedNode(node)
  }, [])

  const onEdgeClick = useCallback((_: React.MouseEvent, edge: Edge<GraphEdgeData>) => {
    setSelectedNode(null)
    setSelectedEdge(edge)
  }, [])

  if (!model) {
    return (
      <p className="text-sm text-[var(--color-muted-foreground)]">
        Сначала постройте BPG.
      </p>
    )
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_300px]">
      <Card className="overflow-hidden">
        <CardContent className="p-0">
          <div className="h-[600px] w-full">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodeClick={onNodeClick}
              onEdgeClick={onEdgeClick}
              fitView
              proOptions={{ hideAttribution: true }}
            >
              <Background />
              <Controls />
              <MiniMap />
            </ReactFlow>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Выбор</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          {!selectedNode && !selectedEdge && (
            <p className="text-[var(--color-muted-foreground)]">
              Граф уровня сущностей: Entity, View, агрегированный cross-view.
            </p>
          )}
          {selectedNode && (
            <>
              <Badge style={{ borderColor: nodeColor(selectedNode.data.nodeKind) }}>
                {selectedNode.data.nodeKind === 'entity' ? 'сущность' : 'экран'}
              </Badge>
              <p className="font-mono text-xs break-all">{selectedNode.id}</p>
              <p>{selectedNode.data.label}</p>
              {selectedNode.data.confidence != null && (
                <p>Уверенность: {selectedNode.data.confidence.toFixed(3)}</p>
              )}
              {selectedNode.data.attributes && (
                <pre className="max-h-48 overflow-auto rounded bg-slate-50 p-2 text-xs">
                  {JSON.stringify(selectedNode.data.attributes, null, 2)}
                </pre>
              )}
            </>
          )}
          {selectedEdge && (
            <>
              <Badge>{selectedEdge.data?.edgeKind}</Badge>
              {selectedEdge.data?.confidence != null && (
                <p>Ср. similarity: {selectedEdge.data.confidence.toFixed(3)}</p>
              )}
              {selectedEdge.data?.metadata && (
                <pre className="max-h-48 overflow-auto rounded bg-slate-50 p-2 text-xs">
                  {JSON.stringify(selectedEdge.data.metadata, null, 2)}
                </pre>
              )}
            </>
          )}
          <div className="border-t pt-2 text-xs text-[var(--color-muted-foreground)]">
            Узлов: {nodes.length} · Рёбер: {edges.length}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
