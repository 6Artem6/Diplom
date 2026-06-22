import { useState } from 'react'
import type { ParsedBbox } from '@/lib/bbox'

export function BboxOverlay({
  src,
  alt,
  bbox,
  stroke = '#2563eb',
  className = 'w-full object-contain',
}: {
  src: string
  alt: string
  bbox: ParsedBbox | null
  stroke?: string
  className?: string
}) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null)

  return (
    <div className="relative overflow-hidden rounded-md border bg-white">
      <img
        src={src}
        alt={alt}
        className={className}
        onLoad={(e) =>
          setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })
        }
      />
      {bbox && size && bbox.width > 0 && bbox.height > 0 && (
        <div
          className="pointer-events-none absolute border-2"
          style={{
            left: `${(bbox.x / size.w) * 100}%`,
            top: `${(bbox.y / size.h) * 100}%`,
            width: `${(bbox.width / size.w) * 100}%`,
            height: `${(bbox.height / size.h) * 100}%`,
            borderColor: stroke,
          }}
        />
      )}
    </div>
  )
}
