import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../api/client";
import type { DocumentTemplate, RenderResponse } from "../api/types";

export function useGlobalTemplates() {
  return useQuery({
    queryKey: ["templates", "global"],
    queryFn: () => apiFetch<DocumentTemplate[]>("/templates"),
  });
}

export function useCreateGlobalTemplate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      name: string;
      description?: string;
      type_tag?: string;
      content: string;
    }) =>
      apiFetch<DocumentTemplate>("/templates", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["templates"] }),
  });
}

export function useUpdateTemplate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      ...body
    }: {
      id: number;
      name?: string;
      description?: string;
      type_tag?: string;
      content?: string;
    }) =>
      apiFetch<DocumentTemplate>(`/templates/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["templates"] }),
  });
}

export function useDeleteTemplate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) =>
      apiFetch(`/templates/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["templates"] }),
  });
}

export function useProjectTemplates(projectSlug: string) {
  return useQuery({
    queryKey: ["templates", "project", projectSlug],
    queryFn: () =>
      apiFetch<DocumentTemplate[]>(
        `/projects/${projectSlug}/templates`
      ),
    enabled: !!projectSlug,
  });
}

export function useCreateProjectTemplate(projectSlug: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      name: string;
      description?: string;
      type_tag?: string;
      content: string;
    }) =>
      apiFetch<DocumentTemplate>(
        `/projects/${projectSlug}/templates`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["templates"] }),
  });
}

export function useRenderTemplate() {
  return useMutation({
    mutationFn: ({
      templateId,
      projectSlug,
      agentName,
    }: {
      templateId: number;
      projectSlug: string;
      agentName?: string;
    }) =>
      apiFetch<RenderResponse>(`/templates/${templateId}/render`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_slug: projectSlug,
          agent_name: agentName || undefined,
        }),
      }),
  });
}
