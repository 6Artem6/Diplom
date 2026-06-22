import type { VisualizationResponse } from '@/api/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogTitle, DialogTrigger } from '@/components/ui/dialog'

export function BpgVisualizationGallery({
  visualization,
}: {
  visualization: VisualizationResponse | null
}) {
  if (!visualization || visualization.visualization_files.length === 0) {
    return null
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Cross-view debug-оверлеи (по экранам)</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {visualization.visualization_files.map((file) => (
            <Dialog key={file.filename}>
              <DialogTrigger asChild>
                <button
                  type="button"
                  className="overflow-hidden rounded-md border bg-white text-left hover:shadow-md"
                >
                  <img
                    src={file.url}
                    alt={file.filename}
                    className="aspect-video w-full object-cover"
                    loading="lazy"
                  />
                  <p className="truncate px-2 py-1 text-xs text-[var(--color-muted-foreground)]">
                    {file.filename}
                  </p>
                </button>
              </DialogTrigger>
              <DialogContent>
                <DialogTitle>{file.filename}</DialogTitle>
                <img
                  src={file.url}
                  alt={file.filename}
                  className="mt-2 max-h-[75vh] w-full object-contain"
                />
              </DialogContent>
            </Dialog>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
