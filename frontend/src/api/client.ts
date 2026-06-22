import axios from 'axios'
import type {
  BuildBpgResponse,
  BusinessProcessGraph,
  VisualizationResponse,
} from './types'

const api = axios.create({
  baseURL: '/api/v1',
  headers: { 'Content-Type': 'application/json' },
})

export async function buildBpg(
  screenshotPaths: string[],
): Promise<BuildBpgResponse> {
  const { data } = await api.post<BuildBpgResponse>('/bpg/build', {
    screenshot_paths: screenshotPaths,
  })
  return data
}

export async function getBpg(bpgId: string): Promise<BusinessProcessGraph> {
  const { data } = await api.get<BusinessProcessGraph>(`/bpg/${bpgId}`)
  return data
}

export async function getVisualization(
  bpgId: string,
): Promise<VisualizationResponse> {
  const { data } = await api.get<VisualizationResponse>(
    `/bpg/${bpgId}/debug/visualization`,
  )
  return data
}

export async function listDebugFormImages(
  folderCandidates: string[],
): Promise<{ folder: string; files: string[] } | null> {
  for (const folder of folderCandidates) {
    try {
      const res = await fetch(`/debug-forms/${encodeURIComponent(folder)}/list`)
      if (!res.ok) continue
      const body = (await res.json()) as { files: string[] }
      if (body.files.length > 0) {
        return { folder, files: body.files }
      }
    } catch {
      // try next candidate
    }
  }
  return null
}

export function debugFormImageUrl(folder: string, filename: string): string {
  return `/debug-forms/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}`
}

export function visualizationImageUrl(url: string): string {
  if (url.startsWith('http')) return url
  return url
}
