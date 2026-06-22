import { useState } from 'react'
import type { NormalizedElement } from '@/lib/bpgNormalizer'
import { demoScreenshotUrl } from '@/lib/bpgNormalizer'

const COLORS = ['#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed', '#0891b2']

export function MultiBboxOverlay({
  screenshot,
  elements,
  selectedId,
  onSelect,
}: {
  screenshot: string
  elements: NormalizedElement[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null)
  const src = demoScreenshotUrl(screenshot)

  return (
    <div className="relative overflow-hidden rounded-md border bg-white">
      <img
        src={src}
        alt={screenshot}
        className="w-full object-contain"
        onLoad={(e) =>
          setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })
        }
      />
      {size &&
        elements.map((el, i) => {
          const bb = el.parsedBbox
          if (!bb || bb.width <= 0 || bb.height <= 0) return null
          const color = COLORS[i % COLORS.length]
          const active = selectedId === el.element_id
          return (
            <button
              key={el.element_id}
              type="button"
              title={`${el.class} (${el.role})`}
              className="absolute border-2 transition-opacity hover:opacity-100"
              style={{
                left: `${(bb.x / size.w) * 100}%`,
                top: `${(bb.y / size.h) * 100}%`,
                width: `${(bb.width / size.w) * 100}%`,
                height: `${(bb.height / size.h) * 100}%`,
                borderColor: color,
                backgroundColor: active ? `${color}33` : 'transparent',
                opacity: active ? 1 : 0.75,
              }}
              onClick={() => onSelect(el.element_id)}
            />
          )
        })}
    </div>
  )
}
