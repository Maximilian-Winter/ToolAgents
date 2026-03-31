import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '../api/client';
import type { Room, Message, PollResponse } from '../api/types';

export function useRooms(slug: string | undefined) {
  return useQuery<Room[]>({
    queryKey: ['rooms', slug],
    queryFn: () => apiFetch<Room[]>(`/projects/${slug}/rooms`),
    enabled: !!slug,
  });
}

export function useCreateRoom(slug: string | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { name: string; topic?: string }) =>
      apiFetch<Room>(`/projects/${slug}/rooms`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['rooms', slug] });
    },
  });
}

export function useMessages(slug: string | undefined, room: string | undefined) {
  return useQuery<Message[]>({
    queryKey: ['messages', slug, room],
    queryFn: () => apiFetch<Message[]>(`/projects/${slug}/rooms/${room}/messages`),
    enabled: !!slug && !!room,
  });
}

export function usePollMessages(slug: string | undefined, room: string | undefined, since: number) {
  return useQuery<PollResponse>({
    queryKey: ['poll', slug, room, since],
    queryFn: () => apiFetch<PollResponse>(`/projects/${slug}/rooms/${room}/poll?since=${since}`),
    enabled: !!slug && !!room && since > 0,
    refetchInterval: 3000,
  });
}

export function useSendMessage(slug: string | undefined, room: string | undefined) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { sender: string; content: string; message_type?: string; reply_to?: number; to?: string }) =>
      apiFetch<Message>(`/projects/${slug}/rooms/${room}/messages`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['messages', slug, room] });
    },
  });
}
