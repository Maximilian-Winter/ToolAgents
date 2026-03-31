import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '../api/client';
import type { Project } from '../api/types';

export function useProjects() {
  return useQuery<Project[]>({
    queryKey: ['projects'],
    queryFn: () => apiFetch<Project[]>('/projects'),
  });
}

export function useProject(slug: string | undefined) {
  return useQuery<Project>({
    queryKey: ['projects', slug],
    queryFn: () => apiFetch<Project>(`/projects/${slug}`),
    enabled: !!slug,
  });
}

export function useCreateProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { name: string; slug?: string; description?: string; working_dir?: string }) =>
      apiFetch<Project>('/projects', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects'] });
    },
  });
}

export function useUpdateProject(slug: string | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { name?: string; description?: string; working_dir?: string }) =>
      apiFetch<Project>(`/projects/${slug}`, {
        method: 'PATCH',
        body: JSON.stringify(data),
      }),
    onSuccess: (project) => {
      qc.invalidateQueries({ queryKey: ['projects'] });
      qc.invalidateQueries({ queryKey: ['projects', slug] });
      if (project && project.slug !== slug) {
        qc.invalidateQueries({ queryKey: ['projects', project.slug] });
      }
    },
  });
}

export function useDeleteProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (slug: string) =>
      apiFetch<null>(`/projects/${slug}`, { method: 'DELETE' }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects'] });
    },
  });
}
