import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '../api/client';
import type { KBDocument, KBDocumentSummary, KBSearchResult, KBTreeNode } from '../api/types';

export function useKBDocuments(slug: string | undefined, prefix?: string, tag?: string) {
  const params = new URLSearchParams();
  if (prefix) params.set('prefix', prefix);
  if (tag) params.set('tag', tag);
  const qs = params.toString();
  return useQuery<KBDocumentSummary[]>({
    queryKey: ['kb-docs', slug, prefix, tag],
    queryFn: () => apiFetch<KBDocumentSummary[]>(`/projects/${slug}/kb${qs ? `?${qs}` : ''}`),
    enabled: !!slug,
  });
}

export function useKBDocument(slug: string | undefined, path: string | undefined) {
  return useQuery<KBDocument>({
    queryKey: ['kb-doc', slug, path],
    queryFn: () => apiFetch<KBDocument>(`/projects/${slug}/kb/${path}`),
    enabled: !!slug && !!path,
  });
}

export function useKBTree(slug: string | undefined) {
  return useQuery<KBTreeNode[]>({
    queryKey: ['kb-tree', slug],
    queryFn: () => apiFetch<KBTreeNode[]>(`/projects/${slug}/kb/tree`),
    enabled: !!slug,
  });
}

export function useKBSearch(slug: string | undefined, query: string, tag?: string) {
  const params = new URLSearchParams({ q: query });
  if (tag) params.set('tag', tag);
  return useQuery<KBSearchResult[]>({
    queryKey: ['kb-search', slug, query, tag],
    queryFn: () => apiFetch<KBSearchResult[]>(`/projects/${slug}/kb/search?${params}`),
    enabled: !!slug && query.length > 0,
  });
}

export function useCreateOrReplaceKBDoc(slug: string | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { path: string; title?: string; tags?: string; content: string; author: string }) =>
      apiFetch<KBDocument>(`/projects/${slug}/kb`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['kb-docs', slug] });
      qc.invalidateQueries({ queryKey: ['kb-tree', slug] });
      qc.invalidateQueries({ queryKey: ['kb-doc', slug] });
    },
  });
}

export function useDeleteKBDoc(slug: string | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (path: string) =>
      apiFetch(`/projects/${slug}/kb/${path}`, { method: 'DELETE' }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['kb-docs', slug] });
      qc.invalidateQueries({ queryKey: ['kb-tree', slug] });
    },
  });
}

export function useMoveKBDoc(slug: string | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ path, newPath }: { path: string; newPath: string }) =>
      apiFetch<KBDocument>(`/projects/${slug}/kb/${path}/move`, {
        method: 'PATCH',
        body: JSON.stringify({ new_path: newPath }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['kb-docs', slug] });
      qc.invalidateQueries({ queryKey: ['kb-tree', slug] });
      qc.invalidateQueries({ queryKey: ['kb-doc', slug] });
    },
  });
}
