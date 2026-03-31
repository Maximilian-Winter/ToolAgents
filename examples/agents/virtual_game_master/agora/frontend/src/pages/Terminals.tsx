import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { useProject } from '../hooks/useProjects';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import { WebLinksAddon } from '@xterm/addon-web-links';
import '@xterm/xterm/css/xterm.css';
import { Button, EmptyState } from '../components/ui';
import { cx } from '../lib/cx';
import styles from './Terminals.module.css';

// ── Output cache (persists across unmount/remount) ──────────
// Stores terminal output so it can be replayed when component remounts
// after navigating away and back.
const outputCache = new Map<string, Uint8Array[]>();
const MAX_CACHE_BYTES = 512 * 1024; // 512KB per session

function cacheOutput(sessionId: string, data: Uint8Array) {
  let chunks = outputCache.get(sessionId);
  if (!chunks) {
    chunks = [];
    outputCache.set(sessionId, chunks);
  }
  chunks.push(new Uint8Array(data)); // copy so the buffer isn't detached
  // Trim if too large — keep the most recent output
  let total = chunks.reduce((sum, c) => sum + c.length, 0);
  while (total > MAX_CACHE_BYTES && chunks.length > 1) {
    total -= chunks.shift()!.length;
  }
}

function replayCache(sessionId: string, term: Terminal) {
  const chunks = outputCache.get(sessionId);
  if (chunks) {
    for (const chunk of chunks) {
      term.write(chunk);
    }
  }
}

function clearCache(sessionId: string) {
  outputCache.delete(sessionId);
}

// ── Session list cache (persists across unmount/remount) ────
let cachedSessions: TerminalInfo[] | null = null;
let cachedActiveId: string | null = null;

// ── Types ───────────────────────────────────────────────────

interface TerminalInfo {
  id: string;
  working_dir: string;
  shell: string;
  mode: string;
  cols: number;
  rows: number;
  created_at: string;
  status: string;
}

type ViewMode = 'focused' | 'grid';

// ── API helpers ─────────────────────────────────────────────

const API = '/api/terminals';

async function apiCreateTerminal(workingDir: string): Promise<TerminalInfo> {
  const res = await fetch(API, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ working_dir: workingDir }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiListTerminals(): Promise<TerminalInfo[]> {
  const res = await fetch(API);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiKillTerminal(id: string): Promise<void> {
  await fetch(`${API}/${id}`, { method: 'DELETE' });
}

function wsUrl(sessionId: string): string {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}/api/terminals/${sessionId}/ws`;
}

// ── SVG icons ───────────────────────────────────────────────

function IconFocused() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="1" y="1" width="12" height="12" rx="2" />
    </svg>
  );
}

function IconGrid() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="1" y="1" width="5" height="5" rx="1" />
      <rect x="8" y="1" width="5" height="5" rx="1" />
      <rect x="1" y="8" width="5" height="5" rx="1" />
      <rect x="8" y="8" width="5" height="5" rx="1" />
    </svg>
  );
}

// ── Terminal Instance Component ─────────────────────────────

function TerminalView({
  session, isActive, viewMode, onSelect, onMaximize, onKill, onExit,
}: {
  session: TerminalInfo;
  isActive: boolean;
  viewMode: ViewMode;
  onSelect: () => void;
  onMaximize: () => void;
  onKill: () => void;
  onExit: () => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<Terminal | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const fitRef = useRef<FitAddon | null>(null);
  const modeRef = useRef<string>(session.mode);
  const lineBufferRef = useRef<string>('');

  const isVisible = viewMode === 'grid' || isActive;

  useEffect(() => {
    if (!containerRef.current) return;

    const term = new Terminal({
      cursorBlink: true,
      fontSize: 13,
      fontFamily: "'Cascadia Code', 'Fira Code', 'JetBrains Mono', Menlo, Monaco, monospace",
      theme: {
        background: '#0d0f14',
        foreground: '#c8cdd8',
        cursor: '#7a8ef7',
        selectionBackground: 'rgba(122,142,247,0.3)',
        black: '#0d0f14', red: '#e06070', green: '#7ecf6a',
        yellow: '#e0a84c', blue: '#7a8ef7', magenta: '#b07af7',
        cyan: '#5cc9d0', white: '#c8cdd8',
        brightBlack: '#4a5070', brightRed: '#e88090', brightGreen: '#98df8a',
        brightYellow: '#f0c87c', brightBlue: '#9aaefc', brightMagenta: '#c89af7',
        brightCyan: '#7cd9e0', brightWhite: '#e8edf8',
      },
      allowProposedApi: true,
    });

    const fitAddon = new FitAddon();
    const webLinksAddon = new WebLinksAddon();
    term.loadAddon(fitAddon);
    term.loadAddon(webLinksAddon);
    term.open(containerRef.current);

    termRef.current = term;
    fitRef.current = fitAddon;

    requestAnimationFrame(() => fitAddon.fit());

    // Replay cached output from previous mount (survives navigation)
    replayCache(session.id, term);

    const ws = new WebSocket(wsUrl(session.id));
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => term.focus();

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const data = new Uint8Array(event.data);
        cacheOutput(session.id, data);
        term.write(data);
      } else if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'connected') {
            modeRef.current = msg.mode || session.mode;
            if (msg.mode === 'pipe') term.write('\x1b[90m[Pipe mode \u2014 local echo enabled]\x1b[0m\r\n');
          } else if (msg.type === 'exited') {
            term.write('\r\n\x1b[90m[Terminal session ended]\x1b[0m\r\n');
            onExit();
          }
        } catch { /* ignore */ }
      }
    };

    ws.onclose = () => term.write('\r\n\x1b[90m[Connection closed]\x1b[0m\r\n');

    term.onData((data) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      if (modeRef.current === 'pipe') {
        for (const ch of data) {
          if (ch === '\r') { term.write('\r\n'); ws.send(new TextEncoder().encode(lineBufferRef.current + '\r\n')); lineBufferRef.current = ''; }
          else if (ch === '\x7f' || ch === '\b') { if (lineBufferRef.current.length > 0) { lineBufferRef.current = lineBufferRef.current.slice(0, -1); term.write('\b \b'); } }
          else if (ch === '\x03') { lineBufferRef.current = ''; term.write('^C\r\n'); ws.send(new TextEncoder().encode('\x03')); }
          else if (ch === '\x04') { ws.send(new TextEncoder().encode('\x04')); }
          else if (ch >= ' ' || ch === '\t') { lineBufferRef.current += ch; term.write(ch); }
        }
      } else {
        ws.send(new TextEncoder().encode(data));
      }
    });

    const handleResize = () => {
      fitAddon.fit();
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
    };

    const resizeObserver = new ResizeObserver(() => requestAnimationFrame(handleResize));
    resizeObserver.observe(containerRef.current);

    return () => { resizeObserver.disconnect(); ws.close(); term.dispose(); termRef.current = null; wsRef.current = null; fitRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.id]);

  useEffect(() => {
    if (isVisible && fitRef.current && termRef.current) {
      const timer = setTimeout(() => { fitRef.current?.fit(); if (isActive) termRef.current?.focus(); }, 60);
      return () => clearTimeout(timer);
    }
  }, [isVisible, isActive, viewMode]);

  const isGrid = viewMode === 'grid';
  const isRunning = session.status === 'running';

  return (
    <div className={isGrid ? (isActive ? styles.gridCellActive : styles.gridCell) : (isActive ? styles.focusedPane : styles.focusedPaneHidden)}>
      {/* Grid header */}
      <div className={styles.gridHeader} style={{ display: isGrid ? 'flex' : 'none' }} onClick={onSelect} onDoubleClick={onMaximize}>
        <div className={styles.gridHeaderLabel}>
          <span className={styles.statusDot} style={{ background: isRunning ? 'var(--accent-green)' : 'var(--text-muted)' }} />
          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
            {session.shell.split(/[/\\]/).pop()} #{session.id}
          </span>
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-muted)' }}>
            {session.working_dir}
          </span>
        </div>
        <div className={styles.gridHeaderActions}>
          <button className={styles.gridMaxBtn} onClick={(e) => { e.stopPropagation(); onMaximize(); }} title="Maximize">&#9634;</button>
          <button className={cx(styles.killBtn)} style={{ color: 'var(--accent-red)' }} onClick={(e) => { e.stopPropagation(); onKill(); }} title="Kill">&times;</button>
        </div>
      </div>

      <div ref={containerRef} className={styles.termContainer} onClick={() => { onSelect(); termRef.current?.focus(); }} />
    </div>
  );
}

// ── Main Terminals Page ─────────────────────────────────────

export default function Terminals() {
  const { slug } = useParams<{ slug: string }>();
  const { data: project } = useProject(slug);
  const [sessions, setSessionsRaw] = useState<TerminalInfo[]>(cachedSessions ?? []);
  const [activeId, setActiveIdRaw] = useState<string | null>(cachedActiveId);
  const [creating, setCreating] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('focused');

  // Wrap setters to also update module-level cache
  const setSessions = useCallback((updater: TerminalInfo[] | ((prev: TerminalInfo[]) => TerminalInfo[])) => {
    setSessionsRaw((prev) => {
      const next = typeof updater === 'function' ? updater(prev) : updater;
      cachedSessions = next;
      return next;
    });
  }, []);

  const setActiveId = useCallback((id: string | null) => {
    setActiveIdRaw(id);
    cachedActiveId = id;
  }, []);

  useEffect(() => {
    apiListTerminals().then((list) => {
      setSessions(list);
      // Clean cache for sessions that no longer exist
      const liveIds = new Set(list.map((s) => s.id));
      for (const cachedId of outputCache.keys()) {
        if (!liveIds.has(cachedId)) clearCache(cachedId);
      }
      if (list.length > 0 && !cachedActiveId) setActiveId(list[0].id);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const workingDir = project?.working_dir || '';

  const handleCreate = useCallback(async () => {
    if (!workingDir || creating) return;
    setCreating(true);
    try {
      const session = await apiCreateTerminal(workingDir);
      setSessions((prev) => [...prev, session]);
      setActiveId(session.id);
    } catch (err) { console.error('Failed to create terminal:', err); }
    finally { setCreating(false); }
  }, [workingDir, creating]);

  const handleKill = useCallback(async (id: string) => {
    await apiKillTerminal(id);
    clearCache(id);
    setSessions((prev) => {
      const remaining = prev.filter((ses) => ses.id !== id);
      if (activeId === id) setActiveId(remaining.length > 0 ? remaining[0].id : null);
      return remaining;
    });
  }, [activeId, setSessions, setActiveId]);

  const handleExit = useCallback(() => { apiListTerminals().then(setSessions); }, []);

  const handleMaximize = useCallback((id: string) => { setActiveId(id); setViewMode('focused'); }, []);

  const activeSession = sessions.find((ses) => ses.id === activeId);

  return (
    <div className={styles.page}>
      {/* Sidebar */}
      <div className={styles.sidebar}>
        <div className={styles.sideHeader}>
          <span className={styles.sideTitle}>Terminals</span>
          <div className={styles.modeToggle}>
            <button className={viewMode === 'focused' ? styles.modeBtnActive : styles.modeBtn} onClick={() => setViewMode('focused')} title="Focused view">
              <IconFocused />
            </button>
            <button className={viewMode === 'grid' ? styles.modeBtnActive : styles.modeBtn} onClick={() => setViewMode('grid')} title="Grid view">
              <IconGrid />
            </button>
          </div>
          <Button size="sm" onClick={handleCreate} disabled={!workingDir || creating}
            title={workingDir ? `New terminal in ${workingDir}` : 'Set working directory first'}>
            {creating ? '...' : '+ New'}
          </Button>
        </div>

        <div className={styles.sessionList}>
          {sessions.length === 0 && (
            <div className={styles.emptyInfo}>
              No terminals open.
              {!workingDir && <div className={styles.emptyWarning}>Set a working directory in project settings first.</div>}
            </div>
          )}

          {sessions.map((session) => {
            const isActive = session.id === activeId;
            const isRunning = session.status === 'running';
            return (
              <div
                key={session.id}
                className={isActive ? styles.sessionItemActive : styles.sessionItem}
                onClick={() => { setActiveId(session.id); }}
                onDoubleClick={() => handleMaximize(session.id)}
              >
                <div className={styles.sessionName}>
                  <span>
                    <span className={styles.statusDot} style={{ background: isRunning ? 'var(--accent-green)' : 'var(--text-muted)' }} />
                    {session.shell.split(/[/\\]/).pop()} #{session.id}
                  </span>
                  <button className={styles.killBtn} onClick={(e) => { e.stopPropagation(); handleKill(session.id); }} title="Kill terminal">
                    &times;
                  </button>
                </div>
                <div className={styles.sessionMeta}>{session.working_dir}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main area */}
      <div className={styles.main}>
        {sessions.length > 0 ? (
          <>
            {viewMode === 'focused' ? (
              <div className={styles.termStack}>
                {sessions.map((session) => (
                  <TerminalView key={session.id} session={session} isActive={session.id === activeId}
                    viewMode="focused" onSelect={() => setActiveId(session.id)}
                    onMaximize={() => handleMaximize(session.id)} onKill={() => handleKill(session.id)} onExit={handleExit} />
                ))}
              </div>
            ) : (
              <div className={styles.termGrid}>
                {sessions.map((session) => (
                  <TerminalView key={session.id} session={session} isActive={session.id === activeId}
                    viewMode="grid" onSelect={() => setActiveId(session.id)}
                    onMaximize={() => handleMaximize(session.id)} onKill={() => handleKill(session.id)} onExit={handleExit} />
                ))}
              </div>
            )}

            {viewMode === 'focused' && activeSession && (
              <div className={styles.statusBar}>
                <span>{activeSession.mode === 'pty' ? 'PTY' : 'PIPE'} &middot; {activeSession.shell} &middot; {activeSession.working_dir}</span>
                <button className={styles.statusKillBtn} onClick={() => handleKill(activeSession.id)}>Kill</button>
              </div>
            )}
          </>
        ) : (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>&#9002;</div>
            <EmptyState
              message="No terminal selected"
              action={workingDir ? (
                <Button onClick={handleCreate} loading={creating}>Create Terminal</Button>
              ) : undefined}
            />
            {!workingDir && (
              <div style={{ fontSize: 12, maxWidth: 280, textAlign: 'center' }}>
                Set a working directory in project settings to enable terminals.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
