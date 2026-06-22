import { useState } from 'react'
import type { BusinessProcessGraph } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface RawJsonTabProps {
  bpg: BusinessProcessGraph | null
  normalizedKeys?: string[]
}

function highlightJson(json: string): string {
  return json
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(
      /("(?:\\.|[^"\\])*")\s*:/g,
      '<span class="json-key">$1</span>:',
    )
    .replace(
      /:\s*("(?:\\.|[^"\\])*")/g,
      ': <span class="json-string">$1</span>',
    )
    .replace(
      /:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)/gi,
      ': <span class="json-number">$1</span>',
    )
    .replace(
      /:\s*(true|false|null)/g,
      ': <span class="json-bool">$1</span>',
    )
}

export function RawJsonTab({ bpg, normalizedKeys }: RawJsonTabProps) {
  const [copied, setCopied] = useState(false)
  const [showKeys, setShowKeys] = useState(false)

  if (!bpg) {
    return (
      <p className="text-sm text-[var(--color-muted-foreground)]">
        Сначала постройте BPG для просмотра raw JSON.
      </p>
    )
  }

  const json = JSON.stringify(bpg, null, 2)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(json)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-4">
          <CardTitle>GET /api/v1/bpg/{'{bpg_id}'}</CardTitle>
          <Button variant="outline" size="sm" onClick={handleCopy}>
            {copied ? 'Скопировано!' : 'Копировать JSON'}
          </Button>
        </CardHeader>
        <CardContent>
          <pre
            className="json-viewer max-h-[70vh] overflow-auto rounded-md border border-slate-200 bg-white p-4 text-left font-mono text-[13px] leading-6 text-slate-800 shadow-inner"
            dangerouslySetInnerHTML={{ __html: highlightJson(json) }}
          />
        </CardContent>
      </Card>

      {normalizedKeys && normalizedKeys.length > 0 && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Найденные ключи JSON (normalizer)</CardTitle>
            <Button variant="outline" size="sm" onClick={() => setShowKeys((v) => !v)}>
              {showKeys ? 'Скрыть' : 'Показать'} ({normalizedKeys.length})
            </Button>
          </CardHeader>
          {showKeys && (
            <CardContent>
              <pre className="max-h-48 overflow-auto rounded bg-slate-50 p-3 text-xs">
                {normalizedKeys.join('\n')}
              </pre>
            </CardContent>
          )}
        </Card>
      )}
    </div>
  )
}
