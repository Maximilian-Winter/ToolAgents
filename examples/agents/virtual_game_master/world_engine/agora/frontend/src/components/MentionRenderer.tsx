import { Link, useParams } from 'react-router-dom';

/**
 * Renders text content with clickable kb: and #N mentions.
 *
 * - kb:path/to/doc.md → green link to /projects/:slug/kb/path/to/doc.md
 * - kb:path/to/doc.md#Section → green link with section
 * - #N → amber link to /projects/:slug/issues/N
 */
export default function MentionRenderer({ text }: { text: string }) {
  const { slug } = useParams<{ slug: string }>();
  if (!slug) return <>{text}</>;

  // Split text by mention patterns, preserving the matches
  const parts: Array<{ type: 'text' | 'kb' | 'issue'; value: string; target?: string }> = [];
  // Combined pattern: kb:... or #N (with lookbehind exclusions handled by checking)
  const pattern = /kb:([^\s]+)|(?<![&#/\w])#(\d+)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    // Add preceding text
    if (match.index > lastIndex) {
      parts.push({ type: 'text', value: text.slice(lastIndex, match.index) });
    }

    if (match[1]) {
      // kb: mention
      parts.push({ type: 'kb', value: `kb:${match[1]}`, target: match[1] });
    } else if (match[2]) {
      // #N mention
      parts.push({ type: 'issue', value: `#${match[2]}`, target: match[2] });
    }

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push({ type: 'text', value: text.slice(lastIndex) });
  }

  if (parts.length === 0) return <>{text}</>;

  return (
    <>
      {parts.map((part, i) => {
        if (part.type === 'kb') {
          const docPath = part.target!.replace(/#.*$/, ''); // strip section for URL
          return (
            <Link
              key={i}
              to={`/projects/${slug}/kb/${docPath}`}
              style={{
                color: '#22c55e',
                background: 'rgba(34, 197, 94, 0.1)',
                padding: '1px 4px',
                borderRadius: '3px',
                textDecoration: 'none',
              }}
            >
              {part.value}
            </Link>
          );
        }
        if (part.type === 'issue') {
          return (
            <Link
              key={i}
              to={`/projects/${slug}/issues/${part.target}`}
              style={{
                color: '#f59e0b',
                background: 'rgba(245, 158, 11, 0.1)',
                padding: '1px 4px',
                borderRadius: '3px',
                textDecoration: 'none',
              }}
            >
              {part.value}
            </Link>
          );
        }
        return <span key={i}>{part.value}</span>;
      })}
    </>
  );
}
