import { useMemo } from 'react'
import type { BusinessProcessGraph, VisualizationResponse } from '@/api/types'
import { normalizeBpg, type NormalizedBpg } from '@/lib/bpgNormalizer'

export function useNormalizedBpg(
  bpg: BusinessProcessGraph | null,
  visualization: VisualizationResponse | null,
): NormalizedBpg | null {
  return useMemo(() => {
    if (!bpg) return null
    return normalizeBpg(bpg, visualization)
  }, [bpg, visualization])
}
