export const MESSAGE_TYPES = [
  'statement', 'proposal', 'objection', 'question', 'answer', 'consensus',
] as const;

export const TYPE_COLORS: Record<string, string> = {
  statement: 'var(--text-secondary)',
  proposal: 'var(--accent-green)',
  objection: 'var(--accent-red)',
  question: 'var(--accent-amber)',
  answer: 'var(--accent-cyan)',
  consensus: 'var(--accent-purple)',
};

const SENDER_COLORS = [
  '#7a8ef7', '#7ecf6a', '#e0a84c', '#e06070', '#b07af7',
  '#5cc9d0', '#f77a7a', '#6ab0cf', '#cf6ab0', '#6acf8f',
];

export function senderColor(name: string): string {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) | 0;
  return SENDER_COLORS[Math.abs(hash) % SENDER_COLORS.length];
}

export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** Check if two messages can be grouped (same sender, within 5 min) */
export function canGroup(a: { sender: string; created_at: string }, b: { sender: string; created_at: string }): boolean {
  if (a.sender !== b.sender) return false;
  const diff = new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
  return diff < 5 * 60 * 1000;
}
