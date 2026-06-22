import type { NormalizedBpg } from '@/lib/bpgNormalizer'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

export function CrossViewTab({ model }: { model: NormalizedBpg | null }) {
  if (!model) {
    return (
      <p className="text-sm text-[var(--color-muted-foreground)]">
        Сначала постройте BPG.
      </p>
    )
  }

  return (
    <div className="space-y-4">
      <div className="rounded-md border border-blue-100 bg-blue-50 px-4 py-3 text-sm text-blue-900">
        Cross-view связи агрегированы на уровне <strong>сущностей</strong>. Технические
        manifestation-пары не отображаются списком.
      </div>

      {model.entityPairLinks.length === 0 ? (
        <Card>
          <CardContent className="py-6 text-sm">
            {model.entities.length === 1 ? (
              <div className="space-y-3">
                <p>
                  Одна сущность — cross-view <strong>внутри сущности</strong> (несколько экранов
                  объединены в один semantic instance).
                </p>
                <EntitySelfMetrics entity={model.entities[0]} />
              </div>
            ) : (
              <p>Агрегированные пары сущностей не найдены.</p>
            )}
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {model.entityPairLinks.map((link) => (
            <Card key={`${link.entity_a_id}-${link.entity_b_id}`}>
              <CardHeader>
                <CardTitle className="flex flex-wrap items-center gap-2 text-base">
                  <span>{link.entity_a_label}</span>
                  <Badge className="bg-orange-100 text-orange-800">↔</Badge>
                  <span>{link.entity_b_label}</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <table className="w-full text-sm">
                  <tbody>
                    <tr>
                      <td className="py-1 text-[var(--color-muted-foreground)]">Связей</td>
                      <td className="py-1 font-medium">{link.count}</td>
                    </tr>
                    <tr>
                      <td className="py-1 text-[var(--color-muted-foreground)]">Ср. similarity</td>
                      <td className="py-1 font-medium">{link.avg_similarity.toFixed(3)}</td>
                    </tr>
                    <tr>
                      <td className="py-1 text-[var(--color-muted-foreground)]">Мин / макс</td>
                      <td className="py-1 font-medium">
                        {link.min_similarity.toFixed(3)} / {link.max_similarity.toFixed(3)}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {model.similarityStats && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Глобальная статистика similarity</CardTitle>
          </CardHeader>
          <CardContent>
            <table className="text-sm">
              <tbody>
                {Object.entries(model.similarityStats).map(([k, v]) => (
                  <tr key={k}>
                    <td className="pr-4 py-1 text-[var(--color-muted-foreground)]">{k}</td>
                    <td>{typeof v === 'number' ? v.toFixed(3) : String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function EntitySelfMetrics({ entity }: { entity: NormalizedBpg['entities'][0] }) {
  const m = entity.metrics
  return (
    <table className="w-full text-sm">
      <tbody>
        <tr>
          <td className="py-1 text-[var(--color-muted-foreground)]">Сущность</td>
          <td className="py-1 font-medium">{entity.type}</td>
        </tr>
        <tr>
          <td className="py-1 text-[var(--color-muted-foreground)]">Экраны</td>
          <td className="py-1">{m.viewsCount}</td>
        </tr>
        <tr>
          <td className="py-1 text-[var(--color-muted-foreground)]">Элементы</td>
          <td className="py-1">{m.representationsCount}</td>
        </tr>
        <tr>
          <td className="py-1 text-[var(--color-muted-foreground)]">Классы</td>
          <td className="py-1">{entity.classes.join(', ') || '—'}</td>
        </tr>
        <tr>
          <td className="py-1 text-[var(--color-muted-foreground)]">Ср. / мин / макс similarity</td>
          <td className="py-1">
            {m.avgSimilarity != null
              ? `${m.avgSimilarity.toFixed(3)} / ${m.minSimilarity?.toFixed(3)} / ${m.maxSimilarity?.toFixed(3)}`
              : 'нет данных'}
          </td>
        </tr>
      </tbody>
    </table>
  )
}
