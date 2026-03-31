import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '../api/client';
import type { ProjectAgent } from '../api/types';

export function useProjectAgents(slug: string | undefined) {
  return useQuery<ProjectAgent[]>({
    queryKey: ['project-agents', slug],
    queryFn: () => apiFetch<ProjectAgent[]>(`/projects/${slug}/agents`),
    enabled: !!slug,
  });
}

export function useAddProjectAgent(slug: string | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      agent_name: string;
      system_prompt?: string;
      initial_task?: string;
      model?: string;
      allowed_tools?: string;
      prompt_source?: string;
      runtime?: string;
      skip_permissions?: boolean;
    }) =>
      apiFetch<ProjectAgent>(`/projects/${slug}/agents`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['project-agents', slug] });
    },
  });
}

export function useUpdateProjectAgent(slug: string | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      agent_name: string;
      system_prompt?: string;
      initial_task?: string;
      model?: string;
      allowed_tools?: string;
      prompt_source?: string;
      skip_permissions?: boolean;
    }) => {
      const { agent_name, ...body } = data;
      return apiFetch<ProjectAgent>(`/projects/${slug}/agents/${agent_name}`, {
        method: 'PATCH',
        body: JSON.stringify(body),
      });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['project-agents', slug] });
    },
  });
}

export function useRemoveProjectAgent(slug: string | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (agentName: string) =>
      apiFetch<null>(`/projects/${slug}/agents/${agentName}`, {
        method: 'DELETE',
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['project-agents', slug] });
    },
  });
}
