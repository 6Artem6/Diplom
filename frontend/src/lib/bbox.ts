import type { DetectedElement } from '@/api/types'

export interface ParsedBbox {
  x: number
  y: number
  width: number
  height: number
}

export function parseBbox(raw: Record<string, number> | null | undefined): ParsedBbox | null {
  if (!raw) return null
  if ('x1' in raw && 'y1' in raw) {
    const x1 = raw.x1
    const y1 = raw.y1
    const x2 = raw.x2 ?? x1 + (raw.width ?? 0)
    const y2 = raw.y2 ?? y1 + (raw.height ?? 0)
    return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 }
  }
  if ('x' in raw && 'y' in raw) {
    return {
      x: raw.x,
      y: raw.y,
      width: raw.width ?? 0,
      height: raw.height ?? 0,
    }
  }
  return null
}

export function elementBbox(el: DetectedElement): ParsedBbox | null {
  return parseBbox(el.bbox)
}
