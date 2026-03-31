import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../api/client';
import type { ProjectStats } from '../api/types';

export function useProjectStats(slug: string | undefined) {
  return useQuery<ProjectStats>({
    queryKey: ['projects', slug, 'stats'],
    queryFn: () => apiFetch<ProjectStats>(`/projects/${slug}/stats`),
    enabled: !!slug,
  });
}
