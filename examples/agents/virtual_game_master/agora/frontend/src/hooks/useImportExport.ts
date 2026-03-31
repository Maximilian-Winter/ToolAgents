import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '../api/client';

interface ImportResponse {
  status: string;
  summary: Record<string, number>;
}

export function useExportProject(slug: string | undefined) {
  return useMutation({
    mutationFn: async () => {
      const res = await fetch(`/api/projects/${slug}/export`);
      if (!res.ok) {
        const error = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(error.detail || res.statusText);
      }
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${slug}-export.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },
  });
}

export function useImportProject(slug: string | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (file: File) => {
      const text = await file.text();
      const json = JSON.parse(text);
      const resp = await apiFetch<ImportResponse>(`/projects/${slug}/import`, {
        method: 'POST',
        body: JSON.stringify(json),
      });
      return resp?.summary ?? {};
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects'] });
      qc.invalidateQueries({ queryKey: ['projects', slug] });
      qc.invalidateQueries({ queryKey: ['issues', slug] });
      qc.invalidateQueries({ queryKey: ['rooms', slug] });
      qc.invalidateQueries({ queryKey: ['agents'] });
      qc.invalidateQueries({ queryKey: ['kb-docs', slug] });
      qc.invalidateQueries({ queryKey: ['kb-tree', slug] });
      qc.invalidateQueries({ queryKey: ['kb-doc', slug] });
      qc.invalidateQueries({ queryKey: ['kb-search', slug] });
    },
  });
}
