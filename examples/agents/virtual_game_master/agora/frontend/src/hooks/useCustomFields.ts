import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../api/client";
import type { CustomFieldDefinition } from "../api/types";

// ── Field Definitions ──

export function useCustomFieldDefinitions(entityType?: "agent" | "project") {
  return useQuery({
    queryKey: ["custom-fields", entityType],
    queryFn: () => {
      const params = entityType ? `?entity_type=${entityType}` : "";
      return apiFetch<CustomFieldDefinition[]>(`/custom-fields${params}`);
    },
  });
}

export function useCreateFieldDefinition() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      name: string;
      label: string;
      field_type: string;
      entity_type: string;
      options_json?: string;
      default_value?: string;
      required?: boolean;
      sort_order?: number;
    }) =>
      apiFetch<CustomFieldDefinition>("/custom-fields", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["custom-fields"] }),
  });
}

export function useUpdateFieldDefinition() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      ...body
    }: {
      id: number;
      label?: string;
      options_json?: string;
      default_value?: string;
      required?: boolean;
      sort_order?: number;
    }) =>
      apiFetch<CustomFieldDefinition>(`/custom-fields/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["custom-fields"] }),
  });
}

export function useDeleteFieldDefinition() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) =>
      apiFetch(`/custom-fields/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["custom-fields"] }),
  });
}

// ── Field Values ──

export function useAgentFields(agentName: string) {
  return useQuery({
    queryKey: ["agent-fields", agentName],
    queryFn: () =>
      apiFetch<Record<string, string>>(`/agents/${agentName}/fields`),
    enabled: !!agentName,
  });
}

export function useSetAgentFields(agentName: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (fields: Record<string, string>) =>
      apiFetch<Record<string, string>>(`/agents/${agentName}/fields`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fields),
      }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["agent-fields", agentName] }),
  });
}

export function useProjectFields(projectSlug: string) {
  return useQuery({
    queryKey: ["project-fields", projectSlug],
    queryFn: () =>
      apiFetch<Record<string, string>>(
        `/projects/${projectSlug}/fields`
      ),
    enabled: !!projectSlug,
  });
}

export function useSetProjectFields(projectSlug: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (fields: Record<string, string>) =>
      apiFetch<Record<string, string>>(
        `/projects/${projectSlug}/fields`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(fields),
        }
      ),
    onSuccess: () =>
      qc.invalidateQueries({
        queryKey: ["project-fields", projectSlug],
      }),
  });
}
