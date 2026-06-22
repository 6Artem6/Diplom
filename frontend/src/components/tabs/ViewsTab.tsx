import { useEffect, useState } from 'react'
import type { NormalizedBpg, NormalizedElement } from '@/lib/bpgNormalizer'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { MultiBboxOverlay } from '@/components/shared/MultiBboxOverlay'

export function ViewsTab({ model }: { model: NormalizedBpg | null }) {
  const [selectedViewId, setSelectedViewId] = useState<string | null>(null)
  const [selectedElId, setSelectedElId] = useState<string | null>(null)

  useEffect(() => {
    if (model?.views.length && !selectedViewId) {
      setSelectedViewId(model.views[0].id)
    }
  }, [model, selectedViewId])

  const view = model?.views.find((v) => v.id === selectedViewId) ?? null
  const element = view?.elements.find((e) => e.element_id === selectedElId) ?? null

  useEffect(() => {
    if (view?.elements.length) {
      setSelectedElId(view.elements[0].element_id)
    } else {
      setSelectedElId(null)
    }
  }, [selectedViewId, view])

  if (!model) {
    return (
      <p className="text-sm text-[var(--color-muted-foreground)]">
        Сначала постройте BPG.
      </p>
    )
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[280px_1fr]">
      <div className="space-y-2">
        <h3 className="text-sm font-medium">Экраны ({model.views.length})</h3>
        {model.views.map((v) => (
          <button
            key={v.id}
            type="button"
            onClick={() => setSelectedViewId(v.id)}
            className={cn(
              'w-full rounded-lg border bg-white p-3 text-left text-sm hover:shadow-sm',
              selectedViewId === v.id && 'border-[var(--color-primary)] ring-1 ring-[var(--color-primary)]',
            )}
          >
            <p className="font-medium">{v.screenshot}</p>
            <p className="text-xs text-[var(--color-muted-foreground)]">
              {v.elements.length} элементов · {v.bboxElements.length} с bbox
            </p>
          </button>
        ))}
      </div>

      {view && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Экран: {view.screenshot}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-xs text-[var(--color-muted-foreground)]">
              view_id: {view.view_uuid ?? 'нет данных'}
            </p>

            <MultiBboxOverlay
              screenshot={view.screenshot}
              elements={view.bboxElements.length ? view.bboxElements : view.elements}
              selectedId={selectedElId}
              onSelect={setSelectedElId}
            />

            {view.bboxElements.length === 0 && view.debugImageUrl && (
              <div className="space-y-1">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  Bbox в API отсутствуют — debug-оверлей:
                </p>
                <img src={view.debugImageUrl} alt="debug" className="w-full rounded border" />
              </div>
            )}

            <div className="max-h-64 overflow-y-auto rounded border">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-slate-100">
                  <tr>
                    <th className="px-2 py-1.5">Класс</th>
                    <th className="px-2 py-1.5">Роль</th>
                    <th className="px-2 py-1.5">Текст</th>
                    <th className="px-2 py-1.5">conf.</th>
                  </tr>
                </thead>
                <tbody>
                  {view.elements.map((el) => (
                    <tr
                      key={el.element_id}
                      className={cn(
                        'cursor-pointer border-b hover:bg-slate-50',
                        selectedElId === el.element_id && 'bg-blue-50',
                      )}
                      onClick={() => setSelectedElId(el.element_id)}
                    >
                      <td className="px-2 py-1.5">
                        <Badge className="text-[10px]">{el.class}</Badge>
                      </td>
                      <td className="px-2 py-1.5">{el.role ?? '—'}</td>
                      <td className="max-w-[140px] truncate px-2 py-1.5">
                        {el.cleaned_text ?? el.text ?? '—'}
                        {el.textIsRawOcr && (
                          <span className="text-[var(--color-muted-foreground)]"> (шумный OCR)</span>
                        )}
                      </td>
                      <td className="px-2 py-1.5">{el.confidence.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {element && <ElementDetail el={element} />}
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function ElementDetail({ el }: { el: NormalizedElement }) {
  return (
    <div className="rounded border bg-slate-50 p-3 text-xs">
      <p className="mb-1 font-medium">Выбранный элемент</p>
      <p>
        <strong>Класс:</strong> {el.class} · <strong>Роль:</strong> {el.role ?? '—'}
      </p>
      <p className="font-mono break-all">{el.element_id}</p>
      <p className="mt-1">
        <strong>Текст:</strong> {el.cleaned_text ?? el.text ?? '—'}
      </p>
      {el.ocr_text && (
        <p>
          <strong>OCR (сырой):</strong> {el.ocr_text}
        </p>
      )}
      {el.embedding_input && (
        <p className="break-all">
          <strong>Embedding input:</strong> {el.embedding_input}
        </p>
      )}
      <p className="mt-1">
        Уверенность: {el.confidence.toFixed(3)} · Стадия: {el.pipeline_stage}
      </p>
      {!el.parsedBbox && (
        <p className="mt-1 text-[var(--color-muted-foreground)]">без геометрии</p>
      )}
    </div>
  )
}
