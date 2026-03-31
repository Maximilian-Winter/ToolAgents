import { useEffect, useRef, useCallback, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import type { Message } from '../api/types';

export interface TypingState {
  [sender: string]: number; // timestamp of last typing event
}

export function useSSE(slug: string | undefined, room: string | undefined) {
  const qc = useQueryClient();
  const esRef = useRef<EventSource | null>(null);
  const [typing, setTyping] = useState<TypingState>({});

  // Clean up stale typing indicators every 4s
  useEffect(() => {
    const interval = setInterval(() => {
      setTyping((prev) => {
        const now = Date.now();
        const next: TypingState = {};
        let changed = false;
        for (const [sender, ts] of Object.entries(prev)) {
          if (now - ts < 5000) {
            next[sender] = ts;
          } else {
            changed = true;
          }
        }
        return changed ? next : prev;
      });
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  const connect = useCallback(() => {
    if (!slug || !room) return;
    if (esRef.current) {
      esRef.current.close();
    }

    const url = `/api/projects/${slug}/rooms/${room}/stream`;
    const es = new EventSource(url);
    esRef.current = es;

    // New message
    es.addEventListener('message', (event) => {
      try {
        const msg: Message = JSON.parse(event.data);
        qc.setQueryData<Message[]>(['messages', slug, room], (old) => {
          if (!old) return [msg];
          if (old.some((m) => m.id === msg.id)) return old;
          return [...old, msg];
        });
        // Clear typing for this sender
        setTyping((prev) => {
          if (msg.sender in prev) {
            const next = { ...prev };
            delete next[msg.sender];
            return next;
          }
          return prev;
        });
      } catch {
        // ignore parse errors
      }
    });

    // Edited message
    es.addEventListener('edit', (event) => {
      try {
        const updated: Message = JSON.parse(event.data);
        qc.setQueryData<Message[]>(['messages', slug, room], (old) => {
          if (!old) return old;
          return old.map((m) => (m.id === updated.id ? updated : m));
        });
      } catch {
        // ignore
      }
    });

    // Reaction added/removed
    es.addEventListener('reaction', (event) => {
      try {
        const data = JSON.parse(event.data);
        // data contains message_id and updated reactions array
        qc.setQueryData<Message[]>(['messages', slug, room], (old) => {
          if (!old) return old;
          return old.map((m) => {
            if (m.id === data.message_id) {
              return { ...m, reactions: data.reactions ?? m.reactions };
            }
            return m;
          });
        });
      } catch {
        // ignore
      }
    });

    // Typing indicator
    es.addEventListener('typing', (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.sender) {
          setTyping((prev) => ({ ...prev, [data.sender]: Date.now() }));
        }
      } catch {
        // ignore
      }
    });

    // Round advance
    es.addEventListener('round', (event) => {
      try {
        const data = JSON.parse(event.data);
        // Invalidate room data to pick up new round number
        qc.invalidateQueries({ queryKey: ['rooms', slug] });
        void data; // consumed for side effect
      } catch {
        // ignore
      }
    });

    // Heartbeat keep-alive
    es.addEventListener('heartbeat', () => {
      // keep-alive, do nothing
    });

    // Legacy ping handler
    es.addEventListener('ping', () => {
      // keep-alive, do nothing
    });

    es.onerror = () => {
      es.close();
      // Reconnect after a delay
      setTimeout(() => connect(), 3000);
    };
  }, [slug, room, qc]);

  useEffect(() => {
    connect();
    return () => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
    };
  }, [connect]);

  return { typing };
}
