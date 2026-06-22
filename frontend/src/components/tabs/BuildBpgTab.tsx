import { useState } from 'react'
import axios from 'axios'
import { buildBpg, getBpg, getVisualization } from '@/api/client'
import type {
  BuildBpgResponse,
  BuildStatus,
  BusinessProcessGraph,
  VisualizationResponse,
} from '@/api/types'
import { toContainerPath } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { StatusSpinner } from '@/components/StatusSpinner'
import { PipelineGallery } from '@/components/PipelineGallery'
import { BpgVisualizationGallery } from '@/components/BpgVisualizationGallery'
import { Badge } from '@/components/ui/badge'

const DEMO_IMAGES = [
  'demo_form_01.png',
  'demo_form_02.png',
  'demo_form_03.png',
  'demo_form_04.png',
  'demo_form_05.png',
  'demo_form_10.png',
  'demo_form_11.png',
  'demo_form_12.png',
  'demo_form_13.png',
  'demo_form_14.png',
  'demo_form_15.png',
  'demo_form_16.png',
  'demo_form_17.png',
  'demo_form_18.png',
]

interface BuildBpgTabProps {
  compact?: boolean
  onBuilt: (data: {
    bpgId: string
    bpg: BusinessProcessGraph
    visualization: VisualizationResponse
    screenshotPaths: string[]
    buildResult: BuildBpgResponse
  }) => void
}

export function BuildBpgTab({ onBuilt, compact = false }: BuildBpgTabProps) {
  const [selected, setSelected] = useState<string[]>(['demo_form_16.png', 'demo_form_11.png'])
  const [status, setStatus] = useState<BuildStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [buildResult, setBuildResult] = useState<BuildBpgResponse | null>(null)
  const [lastPaths, setLastPaths] = useState<string[]>([])
  const [lastVisualization, setLastVisualization] = useState<VisualizationResponse | null>(null)

  const toggle = (filename: string) => {
    setSelected((prev) =>
      prev.includes(filename)
        ? prev.filter((f) => f !== filename)
        : [...prev, filename],
    )
  }

  const handleBuild = async () => {
    if (selected.length === 0) return
    setError(null)
    const paths = selected.map(toContainerPath)
    setLastPaths(paths)

    try {
      setStatus('building')
      const result = await buildBpg(paths)
      setBuildResult(result)

      setStatus('loading_graph')
      const bpg = await getBpg(result.bpg_id)

      setStatus('loading_visualization')
      const visualization = await getVisualization(result.bpg_id)

      setLastVisualization(visualization)
      setStatus('done')
      onBuilt({
        bpgId: result.bpg_id,
        bpg,
        visualization,
        screenshotPaths: paths,
        buildResult: result,
      })
    } catch (e) {
      setStatus('error')
      const msg = axios.isAxiosError(e)
        ? String(e.response?.data?.detail ?? e.message)
        : e instanceof Error
          ? e.message
          : 'Ошибка построения'
      setError(msg)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>{compact ? 'Построение BPG' : 'Выбор скриншотов'}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {!compact && (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Пути отправляются в API как container paths (например{' '}
              <code className="rounded bg-slate-100 px-1">/app/data/demo_forms/images/…</code>
              ).
            </p>
          )}
          <div className={compact ? 'grid max-h-40 grid-cols-2 gap-2 overflow-y-auto sm:grid-cols-4 lg:grid-cols-6' : 'grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3'}>
            {DEMO_IMAGES.map((filename) => (
              <label
                key={filename}
                className="flex cursor-pointer items-center gap-3 rounded-md border bg-white p-3 hover:bg-slate-50"
              >
                <Checkbox
                  checked={selected.includes(filename)}
                  onCheckedChange={() => toggle(filename)}
                />
                <span className="text-sm">{filename}</span>
              </label>
            ))}
          </div>
          <Button onClick={handleBuild} disabled={selected.length === 0 || status === 'building' || status === 'loading_graph' || status === 'loading_visualization'}>
            Построить BPG ({selected.length} скрин.)
          </Button>
          <StatusSpinner status={status} error={error} />
        </CardContent>
      </Card>

      {buildResult && (
        <Card>
          <CardHeader>
            <CardTitle>Результат построения</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            <Badge>BPG ID: {buildResult.bpg_id}</Badge>
            <Badge>Элементов detection: {buildResult.detected_elements_count}</Badge>
            <Badge>Сущностей: {buildResult.entity_instances_count}</Badge>
            <Badge>Cross-view рёбер: {buildResult.cross_view_edges_count}</Badge>
            <Badge>Действий: {buildResult.actions_count}</Badge>
          </CardContent>
        </Card>
      )}

      {!compact && (
        <>
          <BpgVisualizationGallery visualization={lastVisualization} />
          <PipelineGallery screenshotPaths={lastPaths} />
        </>
      )}
    </div>
  )
}
