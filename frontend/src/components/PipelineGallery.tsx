import { useEffect, useState } from 'react'
import {
  debugFormImageUrl,
  listDebugFormImages,
} from '@/api/client'
import { screenshotStem } from '@/lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogTitle, DialogTrigger } from '@/components/ui/dialog'

interface Section {
  stem: string
  folder: string
  files: string[]
}

function folderCandidates(stem: string): string[] {
  return [stem, `${stem}.png`]
}

export function PipelineGallery({ screenshotPaths }: { screenshotPaths: string[] }) {
  const [sections, setSections] = useState<Section[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      const result: Section[] = []
      for (const p of screenshotPaths) {
        const stem = screenshotStem(p)
        const listed = await listDebugFormImages(folderCandidates(stem))
        if (listed) {
          result.push({ stem, folder: listed.folder, files: listed.files })
        }
      }
      if (!cancelled) {
        setSections(result)
        setLoading(false)
      }
    }
    if (screenshotPaths.length > 0) {
      void load()
    } else {
      setSections([])
    }
    return () => {
      cancelled = true
    }
  }, [screenshotPaths])

  if (screenshotPaths.length === 0) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle>Стадии pipeline (debug/forms)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {loading && (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Сканирование папок debug/forms...
          </p>
        )}
        {!loading && sections.length === 0 && (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Изображения pipeline не найдены в debug/forms для выбранных скриншотов.
          </p>
        )}
        {sections.map((section) => (
          <div key={section.stem} className="space-y-3">
            <h4 className="font-medium">{section.stem}</h4>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
              {section.files.map((file) => {
                const src = debugFormImageUrl(section.folder, file)
                return (
                  <Dialog key={file}>
                    <DialogTrigger asChild>
                      <button
                        type="button"
                        className="group overflow-hidden rounded-md border bg-white text-left transition hover:shadow-md"
                      >
                        <img
                          src={src}
                          alt={file}
                          className="aspect-video w-full object-cover object-top"
                          loading="lazy"
                        />
                        <p className="truncate px-2 py-1.5 text-xs text-[var(--color-muted-foreground)] group-hover:text-[var(--color-foreground)]">
                          {file}
                        </p>
                      </button>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogTitle>{section.stem} — {file}</DialogTitle>
                      <img
                        src={src}
                        alt={file}
                        className="mt-2 max-h-[75vh] w-full object-contain"
                      />
                    </DialogContent>
                  </Dialog>
                )
              })}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
