import { Loader2 } from 'lucide-react'
import type { BuildStatus } from '@/api/types'

const STATUS_LABELS: Record<BuildStatus, string> = {
  idle: '',
  building: 'Построение BPG...',
  loading_graph: 'Загрузка графа...',
  loading_visualization: 'Загрузка визуализации...',
  done: 'Готово',
  error: 'Ошибка',
}

export function StatusSpinner({
  status,
  error,
}: {
  status: BuildStatus
  error?: string | null
}) {
  if (status === 'idle') return null

  return (
    <div className="flex items-center gap-2 rounded-md border bg-white px-4 py-3 text-sm">
      {status !== 'done' && status !== 'error' && (
        <Loader2 className="h-4 w-4 animate-spin text-[var(--color-primary)]" />
      )}
      <span>{STATUS_LABELS[status]}</span>
      {error && <span className="text-red-600">— {error}</span>}
    </div>
  )
}
