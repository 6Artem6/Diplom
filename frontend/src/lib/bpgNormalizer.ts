import type {
  BusinessProcessGraph,
  DetectedElement,
  VisualizationResponse,
} from '@/api/types'
import { elementBbox, type ParsedBbox } from '@/lib/bbox'

export interface NormalizedElement extends DetectedElement {
  parsedBbox: ParsedBbox | null
  textIsRawOcr: boolean
}

export interface NormalizedView {
  id: string
  screenshot: string
  view_uuid: string | null
  debugImageUrl: string | null
  elements: NormalizedElement[]
  bboxElements: NormalizedElement[]
}

export interface EntityMetrics {
  representationsCount: number
  viewsCount: number
  confidence: number
  avgSimilarity: number | null
  minSimilarity: number | null
  maxSimilarity: number | null
  crossViewEdgeCount: number
}

export interface NormalizedEntity {
  id: string
  type: string
  confidence: number
  views: NormalizedView[]
  elements: NormalizedElement[]
  roles: string[]
  texts: string[]
  classes: string[]
  bboxes: ParsedBbox[]
  evidence: string[]
  metrics: EntityMetrics
  vizColor: string | null
  isCrossView: boolean
  attributes: Record<string, unknown>
}

export interface EntityPairLink {
  entity_a_id: string
  entity_a_label: string
  entity_b_id: string
  entity_b_label: string
  avg_similarity: number
  min_similarity: number
  max_similarity: number
  count: number
}

export interface NormalizedBpg {
  bpg_id: string
  entities: NormalizedEntity[]
  views: NormalizedView[]
  entityPairLinks: EntityPairLink[]
  allKeysFound: string[]
  similarityStats: Record<string, number | string> | null
  summary: Record<string, unknown> | null
  detectedElements: NormalizedElement[]
}

export function collectAllKeys(
  obj: unknown,
  prefix = '',
  out: Set<string> = new Set(),
): Set<string> {
  if (obj === null || obj === undefined) return out
  if (Array.isArray(obj)) {
    if (obj.length > 0) collectAllKeys(obj[0], `${prefix}[]`, out)
    return out
  }
  if (typeof obj === 'object') {
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      const path = prefix ? `${prefix}.${k}` : k
      out.add(path)
      collectAllKeys(v, path, out)
    }
  }
  return out
}

function normalizeElement(el: DetectedElement): NormalizedElement {
  return {
    ...el,
    parsedBbox: elementBbox(el),
    textIsRawOcr: Boolean(el.noisy_ocr),
  }
}

function getDetectedElements(
  bpg: BusinessProcessGraph,
  visualization: VisualizationResponse | null,
): DetectedElement[] {
  if (bpg.detected_elements?.length) return bpg.detected_elements
  if (visualization?.detected_elements?.length) return visualization.detected_elements
  return []
}

function buildViews(
  elements: NormalizedElement[],
  visualization: VisualizationResponse | null,
): NormalizedView[] {
  const byScreenshot = new Map<string, NormalizedElement[]>()
  const viewUuidByShot = new Map<string, string>()

  for (const el of elements) {
    const shot = el.screenshot_id
    if (!byScreenshot.has(shot)) byScreenshot.set(shot, [])
    byScreenshot.get(shot)!.push(el)
    if (el.view_id) viewUuidByShot.set(shot, el.view_id)
  }

  return [...byScreenshot.entries()]
    .map(([screenshot, els]) => {
      const viewUuid = viewUuidByShot.get(screenshot) ?? els[0]?.view_id ?? null
      const debugFile = visualization?.visualization_files.find(
        (f) => viewUuid && (f.filename === `${viewUuid}.png` || f.filename.startsWith(viewUuid)),
      )
      return {
        id: viewUuid ?? screenshot,
        screenshot,
        view_uuid: viewUuid,
        debugImageUrl: debugFile?.url ?? null,
        elements: els,
        bboxElements: els.filter((e) => e.parsedBbox != null),
      }
    })
    .sort((a, b) => a.screenshot.localeCompare(b.screenshot))
}

function entityMetrics(
  entity: BusinessProcessGraph['entity_instances'][0],
  elements: NormalizedElement[],
  bpg: BusinessProcessGraph,
): EntityMetrics {
  const elIds = new Set(elements.map((e) => e.element_id))
  const edges = bpg.cross_view_edges.filter(
    (e) => elIds.has(e.source_id) || elIds.has(e.target_id),
  )
  const scores = edges.map((e) => e.similarity_score).filter((s): s is number => s != null)
  const viewIds = new Set(elements.map((e) => e.view_id))

  return {
    representationsCount: elements.length,
    viewsCount:
      viewIds.size ||
      Number(entity.attributes.view_count) ||
      new Set(elements.map((e) => e.screenshot_id)).size,
    confidence: entity.confidence.score,
    avgSimilarity: scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null,
    minSimilarity: scores.length ? Math.min(...scores) : null,
    maxSimilarity: scores.length ? Math.max(...scores) : null,
    crossViewEdgeCount: edges.length,
  }
}

function buildEntityPairLinks(
  bpg: BusinessProcessGraph,
  entities: NormalizedEntity[],
): EntityPairLink[] {
  const elToEntity = new Map<string, string>()
  for (const ent of entities) {
    for (const el of ent.elements) {
      elToEntity.set(el.element_id, ent.id)
    }
  }

  const pairScores = new Map<string, number[]>()
  for (const edge of bpg.cross_view_edges) {
    const ea = elToEntity.get(edge.source_id)
    const eb = elToEntity.get(edge.target_id)
    if (!ea || !eb || ea === eb) continue
    const key = [ea, eb].sort().join('::')
    if (!pairScores.has(key)) pairScores.set(key, [])
    if (edge.similarity_score != null) pairScores.get(key)!.push(edge.similarity_score)
  }

  const label = (id: string) => entities.find((e) => e.id === id)?.type ?? id.slice(0, 8)

  return [...pairScores.entries()].map(([key, scores]) => {
    const [a, b] = key.split('::')
    return {
      entity_a_id: a,
      entity_a_label: label(a),
      entity_b_id: b,
      entity_b_label: label(b),
      avg_similarity: scores.reduce((x, y) => x + y, 0) / scores.length,
      min_similarity: Math.min(...scores),
      max_similarity: Math.max(...scores),
      count: scores.length,
    }
  })
}

export function normalizeBpg(
  bpg: BusinessProcessGraph,
  visualization: VisualizationResponse | null,
): NormalizedBpg {
  const allKeysFound = [
    ...collectAllKeys(bpg),
    ...(visualization ? collectAllKeys(visualization) : []),
  ].sort()

  if (import.meta.env.DEV) {
    console.log('[bpgNormalizer] all_keys_found:', allKeysFound)
  }

  const rawElements = getDetectedElements(bpg, visualization)
  const detectedElements = rawElements.map(normalizeElement)

  const typeNames = new Map(bpg.entity_types.map((t) => [t.id, t.name]))
  const vizByEntity = new Map(
    (visualization?.entity_instances ?? []).map((e) => [e.id, e]),
  )
  const allViews = buildViews(detectedElements, visualization)

  const entities: NormalizedEntity[] = bpg.entity_instances.map((entity) => {
    const elementIds = new Set(
      (entity.attributes.element_ids as string[] | undefined)?.map(String) ?? [],
    )
    const elements = detectedElements.filter(
      (el) =>
        el.entity_instance_id === entity.id ||
        elementIds.has(el.element_id) ||
        elementIds.has(String(el.element_id)),
    )

    const viewScreens = new Set(elements.map((e) => e.screenshot_id))
    const entityViews = allViews.filter((v) => viewScreens.has(v.screenshot))

    const roles = [...new Set(elements.map((e) => e.role).filter(Boolean) as string[])]
    const classes = [...new Set(elements.map((e) => e.class))]
    const texts = [
      ...new Set(
        elements
          .map((e) => e.text ?? e.ocr_text)
          .filter((t): t is string => !!t),
      ),
    ]
    const bboxes = elements.map((e) => e.parsedBbox).filter((b): b is ParsedBbox => b != null)
    const evidence = [...new Set(elements.map((e) => e.screenshot_id))]

    const viz = vizByEntity.get(entity.id)
    const metrics = entityMetrics(entity, elements, bpg)

    return {
      id: entity.id,
      type: typeNames.get(entity.entity_type_id) ?? entity.entity_type_id,
      confidence: entity.confidence.score,
      views: entityViews,
      elements,
      roles: roles.length ? roles : ['нет данных'],
      texts,
      classes,
      bboxes,
      evidence: evidence.length ? evidence : [...new Set(entity.provenance.evidence_sources)],
      metrics,
      vizColor: viz?.color ?? null,
      isCrossView: viz?.is_cross_view ?? metrics.viewsCount >= 2,
      attributes: entity.attributes,
    }
  })

  return {
    bpg_id: bpg.id,
    entities,
    views: allViews,
    entityPairLinks: buildEntityPairLinks(bpg, entities),
    allKeysFound,
    similarityStats:
      visualization?.similarity_stats && 'count' in visualization.similarity_stats
        ? (visualization.similarity_stats as Record<string, number | string>)
        : null,
    summary: visualization?.summary
      ? (visualization.summary as unknown as Record<string, unknown>)
      : null,
    detectedElements,
  }
}

export function demoScreenshotUrl(screenshot: string): string {
  const name = screenshot.endsWith('.png') ? screenshot : `${screenshot}.png`
  return `/demo-screenshots/${encodeURIComponent(name)}`
}
