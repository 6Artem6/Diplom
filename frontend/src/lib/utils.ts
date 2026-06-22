import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function screenshotStem(path: string): string {
  const name = path.split('/').pop() ?? path
  return name.replace(/\.[^.]+$/, '')
}

export function toContainerPath(filename: string): string {
  return `/app/data/demo_forms/images/${filename}`
}
