import { useState } from 'react'
import type {
  BuildBpgResponse,
  BusinessProcessGraph,
  VisualizationResponse,
} from '@/api/types'
import { useNormalizedBpg } from '@/hooks/useNormalizedBpg'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { BuildBpgTab } from '@/components/tabs/BuildBpgTab'
import { EntitiesTab } from '@/components/tabs/EntitiesTab'
import { ViewsTab } from '@/components/tabs/ViewsTab'
import { CrossViewTab } from '@/components/tabs/CrossViewTab'
import { GraphTab } from '@/components/tabs/GraphTab'
import { RawJsonTab } from '@/components/tabs/RawJsonTab'
import { Badge } from '@/components/ui/badge'

export default function App() {
  const [bpgId, setBpgId] = useState<string | null>(null)
  const [bpg, setBpg] = useState<BusinessProcessGraph | null>(null)
  const [visualization, setVisualization] = useState<VisualizationResponse | null>(null)
  const [, setBuildResult] = useState<BuildBpgResponse | null>(null)
  const [mainTab, setMainTab] = useState('entities')

  const normalized = useNormalizedBpg(bpg, visualization)

  const handleBuilt = (data: {
    bpgId: string
    bpg: BusinessProcessGraph
    visualization: VisualizationResponse
    buildResult: BuildBpgResponse
  }) => {
    setBpgId(data.bpgId)
    setBpg(data.bpg)
    setVisualization(data.visualization)
    setBuildResult(data.buildResult)
    setMainTab('entities')
  }

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">
              BPG Demo — Business Process Graph
            </h1>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Детекция → сущности → cross-view → граф знаний
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {bpgId && (
              <Badge className="font-mono text-xs">BPG {bpgId.slice(0, 8)}…</Badge>
            )}
            {normalized && (
              <Badge className="text-xs">
                {normalized.detectedElements.length} элементов detection
              </Badge>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-4 py-6">
        <BuildBpgTab onBuilt={handleBuilt} compact />

        <Tabs value={mainTab} onValueChange={setMainTab}>
          <TabsList className="flex h-auto w-full flex-wrap justify-start gap-1">
            <TabsTrigger value="entities">Сущности</TabsTrigger>
            <TabsTrigger value="views">Экраны</TabsTrigger>
            <TabsTrigger value="cross-view">Cross-View</TabsTrigger>
            <TabsTrigger value="graph">Граф</TabsTrigger>
            <TabsTrigger value="raw">Raw JSON</TabsTrigger>
          </TabsList>

          <TabsContent value="entities">
            <EntitiesTab model={normalized} />
          </TabsContent>
          <TabsContent value="views">
            <ViewsTab model={normalized} />
          </TabsContent>
          <TabsContent value="cross-view">
            <CrossViewTab model={normalized} />
          </TabsContent>
          <TabsContent value="graph">
            <GraphTab model={normalized} />
          </TabsContent>
          <TabsContent value="raw">
            <RawJsonTab bpg={bpg} normalizedKeys={normalized?.allKeysFound} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
