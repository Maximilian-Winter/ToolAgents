import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../api/client';
import type { MentionRef } from '../api/types';

export function useKBMentions(slug: string | undefined, kbPath: string | undefined) {
  return useQuery<MentionRef[]>({
    queryKey: ['mentions', slug, 'kb', kbPath],
    queryFn: () => apiFetch<MentionRef[]>(`/projects/${slug}/mentions?kb_path=${encodeURIComponent(kbPath!)}`),
    enabled: !!slug && !!kbPath,
  });
}

export function useIssueMentions(slug: string | undefined, issueNumber: number | undefined) {
  return useQuery<MentionRef[]>({
    queryKey: ['mentions', slug, 'issue', issueNumber],
    queryFn: () => apiFetch<MentionRef[]>(`/projects/${slug}/mentions?issue_number=${issueNumber}`),
    enabled: !!slug && issueNumber !== undefined,
  });
}
