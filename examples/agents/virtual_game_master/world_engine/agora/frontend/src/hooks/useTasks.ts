import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '../api/client';
import type { Issue, Comment } from '../api/types';

export function useIssues(slug: string | undefined, filters?: { state?: string; priority?: string; assignee?: string }) {
  const params = new URLSearchParams();
  if (filters?.state) params.set('state', filters.state);
  if (filters?.priority) params.set('priority', filters.priority);
  if (filters?.assignee) params.set('assignee', filters.assignee);
  const qs = params.toString();
  return useQuery<Issue[]>({
    queryKey: ['issues', slug, filters],
    queryFn: () => apiFetch<Issue[]>(`/projects/${slug}/issues${qs ? `?${qs}` : ''}`),
    enabled: !!slug,
  });
}

export function useIssue(slug: string | undefined, number: number | undefined) {
  return useQuery<Issue>({
    queryKey: ['issue', slug, number],
    queryFn: () => apiFetch<Issue>(`/projects/${slug}/issues/${number}`),
    enabled: !!slug && number !== undefined,
  });
}

export function useCreateIssue(slug: string | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { title: string; body?: string; priority?: string; assignee?: string; reporter: string; labels?: string[] }) =>
      apiFetch<Issue>(`/projects/${slug}/issues`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['issues', slug] });
    },
  });
}

export function useUpdateIssue(slug: string | undefined, number: number | undefined, actor = 'user') {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { state?: string; priority?: string; assignee?: string; title?: string; body?: string }) =>
      apiFetch<Issue>(`/projects/${slug}/issues/${number}?actor=${encodeURIComponent(actor)}`, {
        method: 'PATCH',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['issues', slug] });
      qc.invalidateQueries({ queryKey: ['issue', slug, number] });
    },
  });
}

export function useComments(slug: string | undefined, number: number | undefined) {
  return useQuery<Comment[]>({
    queryKey: ['comments', slug, number],
    queryFn: () => apiFetch<Comment[]>(`/projects/${slug}/issues/${number}/comments`),
    enabled: !!slug && number !== undefined,
  });
}

export function useCreateComment(slug: string | undefined, number: number | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { author: string; body: string }) =>
      apiFetch<Comment>(`/projects/${slug}/issues/${number}/comments`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['comments', slug, number] });
      qc.invalidateQueries({ queryKey: ['issue', slug, number] });
    },
  });
}
