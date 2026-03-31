import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '../api/client';
import type { Agent } from '../api/types';

export function useAgents() {
  return useQuery<Agent[]>({
    queryKey: ['agents'],
    queryFn: () => apiFetch<Agent[]>('/agents'),
  });
}

export function useCreateAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { name: string; display_name?: string; role?: string; token?: string }) =>
      apiFetch<Agent>('/agents', { method: 'POST', body: JSON.stringify(data) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['agents'] }); },
  });
}

export function useUpdateAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { name: string; display_name?: string; role?: string }) =>
      apiFetch<Agent>(`/agents/${data.name}`, {
        method: 'PATCH',
        body: JSON.stringify({ display_name: data.display_name, role: data.role }),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['agents'] }); },
  });
}

export function useDeleteAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch<null>(`/agents/${name}`, { method: 'DELETE' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['agents'] }); },
  });
}
