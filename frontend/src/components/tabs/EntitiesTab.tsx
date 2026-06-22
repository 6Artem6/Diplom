import { useEffect, useState } from 'react'
import type { NormalizedBpg, NormalizedElement } from '@/lib/bpgNormalizer'
import { demoScreenshotUrl } from '@/lib/bpgNormalizer'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { BboxOverlay } from '@/components/shared/BboxOverlay'

export function EntitiesTab({ model }: { model: NormalizedBpg | null }) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedElId, setSelectedElId] = useState<string | null>(null)

  useEffect(() => {
    if (model?.entities.length && !selectedId) {
      setSelectedId(model.entities[0].id)
    }
  }, [model, selectedId])

  const entity = model?.entities.find((e) => e.id === selectedId) ?? null
  const element = entity?.elements.find((e) => e.element_id === selectedElId) ?? null

  useEffect(() => {
    if (entity?.elements.length) {
      setSelectedElId(entity.elements[0].element_id)
    } else {
      setSelectedElId(null)
    }
  }, [selectedId, entity])

  if (!model) {
    return (
      <p className="text-sm text-[var(--color-muted-foreground)]">
        Сначала постройте BPG (панель выше).
      </p>
    )
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[320px_1fr]">
      <div className="space-y-2">
        <h3 className="text-sm font-medium">Сущности ({model.entities.length})</h3>
        {model.entities.map((e) => (
          <button
            key={e.id}
            type="button"
            onClick={() => setSelectedId(e.id)}
            className={cn(
              'w-full rounded-lg border bg-white p-4 text-left hover:shadow-sm',
              selectedId === e.id && 'border-[var(--color-primary)] ring-1 ring-[var(--color-primary)]',
            )}
          >
            <div className="mb-2 flex flex-wrap gap-1">
              <span className="font-mono text-xs">{e.id.slice(0, 8)}…</span>
              {e.isCrossView && (
                <Badge className="bg-orange-100 text-orange-800">cross-view</Badge>
              )}
            </div>
            <dl className="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
              <dt className="text-[var(--color-muted-foreground)]">Тип</dt>
              <dd>{e.type}</dd>
              <dt className="text-[var(--color-muted-foreground)]">Классы</dt>
              <dd className="truncate">{e.classes.slice(0, 3).join(', ') || '—'}</dd>
              <dt className="text-[var(--color-muted-foreground)]">Уверенность</dt>
              <dd>{e.confidence.toFixed(3)}</dd>
              <dt className="text-[var(--color-muted-foreground)]">Экраны</dt>
              <dd>{e.metrics.viewsCount}</dd>
              <dt className="text-[var(--color-muted-foreground)]">Элементы</dt>
              <dd>{e.metrics.representationsCount}</dd>
            </dl>
          </button>
        ))}
      </div>

      {entity ? (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Сущность ↔ элементы</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <p className="font-mono text-xs break-all text-[var(--color-muted-foreground)]">
                {entity.id}
              </p>
              <MetricsGrid metrics={entity.metrics} />
              <FieldRow label="Роли" value={entity.roles.join(', ')} />
              <FieldRow label="Классы (detection)" value={entity.classes.join(', ') || 'нет данных'} />
              <FieldRow label="Источники" value={entity.evidence.join(', ') || 'нет данных'} />

              <div className="grid gap-4 lg:grid-cols-2">
                <div>
                  <h4 className="mb-2 font-medium">
                    Элементы ({entity.elements.length})
                  </h4>
                  <div className="max-h-72 overflow-y-auto rounded border">
                    <table className="w-full text-left text-xs">
                      <thead className="sticky top-0 bg-slate-100">
                        <tr>
                          <th className="px-2 py-1.5">Класс</th>
                          <th className="px-2 py-1.5">Роль</th>
                          <th className="px-2 py-1.5">Экран</th>
                          <th className="px-2 py-1.5">conf.</th>
                        </tr>
                      </thead>
                      <tbody>
                        {entity.elements.map((el) => (
                          <ElementRow
                            key={el.element_id}
                            el={el}
                            selected={selectedElId === el.element_id}
                            onSelect={() => setSelectedElId(el.element_id)}
                          />
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                <div>
                  <h4 className="mb-2 font-medium">Просмотр элемента</h4>
                  <ElementInspector el={element} entityColor={entity.vizColor} />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      ) : (
        <p className="text-sm text-[var(--color-muted-foreground)]">Выберите сущность</p>
      )}
    </div>
  )
}

function ElementInspector({
  el,
  entityColor,
}: {
  el: NormalizedElement | null
  entityColor: string | null
}) {
  if (!el) {
    return (
      <div className="flex h-48 items-center justify-center rounded border bg-slate-50 text-xs text-[var(--color-muted-foreground)]">
        Выберите элемент
      </div>
    )
  }

  const screenshotSrc = demoScreenshotUrl(el.screenshot_id)

  return (
    <div className="space-y-2 text-xs">
      {el.parsedBbox ? (
        <BboxOverlay
          src={screenshotSrc}
          alt={el.screenshot_id}
          bbox={el.parsedBbox}
          stroke={entityColor ?? undefined}
        />
      ) : (
        <div className="rounded border bg-slate-50 p-3 text-[var(--color-muted-foreground)]">
          Элемент без геометрии (bbox отсутствует в API)
        </div>
      )}
      <dl className="space-y-1 rounded border bg-slate-50 p-3">
        <Field label="element_id" value={el.element_id} />
        <Field label="view_id" value={el.view_id} />
        <Field label="Класс" value={el.class} />
        <Field label="Роль" value={el.role ?? 'нет данных'} />
        <Field
          label="Текст (очищенный)"
          value={
            el.cleaned_text ?? el.text
              ? `${el.cleaned_text ?? el.text}${el.noisy_ocr ? ' (шумный OCR)' : ''}`
              : 'нет данных'
          }
        />
        <Field label="OCR (сырой)" value={el.ocr_text ?? 'нет данных'} />
        <Field label="Embedding input" value={el.embedding_input ?? 'нет данных'} />
        <Field label="Уверенность" value={el.confidence.toFixed(3)} />
        <Field label="Стадия" value={el.pipeline_stage} />
      </dl>
    </div>
  )
}

function ElementRow({
  el,
  selected,
  onSelect,
}: {
  el: NormalizedElement
  selected: boolean
  onSelect: () => void
}) {
  return (
    <tr
      className={cn('cursor-pointer border-b hover:bg-slate-50', selected && 'bg-blue-50')}
      onClick={onSelect}
    >
      <td className="px-2 py-1.5 font-medium">{el.class}</td>
      <td className="px-2 py-1.5">{el.role ?? '—'}</td>
      <td className="px-2 py-1.5">{el.screenshot_id}</td>
      <td className="px-2 py-1.5">{el.confidence.toFixed(2)}</td>
    </tr>
  )
}

function MetricsGrid({ metrics }: { metrics: NormalizedBpg['entities'][0]['metrics'] }) {
  return (
    <table className="w-full text-sm">
      <tbody>
        <tr className="border-b">
          <td className="py-1.5 text-[var(--color-muted-foreground)]">Элементы</td>
          <td className="py-1.5 font-medium">{metrics.representationsCount}</td>
          <td className="py-1.5 text-[var(--color-muted-foreground)]">Экраны</td>
          <td className="py-1.5 font-medium">{metrics.viewsCount}</td>
        </tr>
        <tr className="border-b">
          <td className="py-1.5 text-[var(--color-muted-foreground)]">Уверенность</td>
          <td className="py-1.5 font-medium">{metrics.confidence.toFixed(3)}</td>
          <td className="py-1.5 text-[var(--color-muted-foreground)]">Ср. similarity</td>
          <td className="py-1.5 font-medium">
            {metrics.avgSimilarity != null ? metrics.avgSimilarity.toFixed(3) : 'нет данных'}
          </td>
        </tr>
      </tbody>
    </table>
  )
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[100px_1fr] gap-2">
      <dt className="text-[var(--color-muted-foreground)]">{label}</dt>
      <dd className="break-all font-mono">{value}</dd>
    </div>
  )
}

function FieldRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-[var(--color-muted-foreground)]">{label}: </span>
      <span>{value}</span>
    </div>
  )
}
