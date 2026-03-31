import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useKBTree, useKBDocument, useKBSearch, useDeleteKBDoc } from '../hooks/useKnowledgeBase';
import { useKBMentions } from '../hooks/useMentions';
import MentionRenderer from '../components/MentionRenderer';
import type { KBTreeNode } from '../api/types';
import styles from './KnowledgeBase.module.css';

function TreeNode({ node, selectedPath, onSelect }: {
  node: KBTreeNode;
  selectedPath: string | null;
  onSelect: (path: string) => void;
}) {
  const [expanded, setExpanded] = useState(true);

  if (node.children) {
    return (
      <div>
        <div className={styles.treeDir} onClick={() => setExpanded(!expanded)}>
          <span>{expanded ? '▾' : '▸'}</span> {node.name}/
        </div>
        {expanded && (
          <div className={styles.treeChildren}>
            {node.children.map((child) => (
              <TreeNode key={child.name} node={child} selectedPath={selectedPath} onSelect={onSelect} />
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      className={`${styles.treeFile} ${node.path === selectedPath ? styles.treeFileActive : ''}`}
      onClick={() => node.path && onSelect(node.path)}
    >
      {node.name}
    </div>
  );
}

export default function KnowledgeBase() {
  const { slug } = useParams<{ slug: string }>();
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTag, setActiveTag] = useState<string | null>(null);
  const [showRefs, setShowRefs] = useState(false);

  const { data: tree } = useKBTree(slug);
  const { data: doc } = useKBDocument(slug, selectedPath ?? undefined);
  const { data: searchResults } = useKBSearch(slug, searchQuery, activeTag ?? undefined);
  const { data: mentions } = useKBMentions(slug, selectedPath ?? undefined);
  const deleteMut = useDeleteKBDoc(slug);

  const handleDelete = () => {
    if (!selectedPath) return;
    if (!confirm(`Delete ${selectedPath}?`)) return;
    deleteMut.mutate(selectedPath, {
      onSuccess: () => setSelectedPath(null),
    });
  };

  return (
    <div className={styles.container}>
      {/* Left: Tree Sidebar */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <span className={styles.sidebarTitle}>Documents</span>
          <Link to={`/projects/${slug}/kb/new`} className={styles.newBtn}>+ New</Link>
        </div>
        <input
          className={styles.searchInput}
          placeholder="Search documents..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        {activeTag && (
          <div className={styles.activeTagBar}>
            <span className={styles.tagChipActive}>
              {activeTag} <span onClick={() => setActiveTag(null)} style={{ cursor: 'pointer' }}>×</span>
            </span>
          </div>
        )}

        {searchQuery ? (
          <div className={styles.searchResults}>
            {searchResults?.map((r) => (
              <div key={r.path} className={styles.treeFile} onClick={() => { setSelectedPath(r.path); setSearchQuery(''); }}>
                <div className={styles.searchResultTitle}>{r.title}</div>
                <div className={styles.searchResultSnippet} dangerouslySetInnerHTML={{ __html: r.snippet }} />
              </div>
            ))}
            {searchResults?.length === 0 && <div className={styles.empty}>No results</div>}
          </div>
        ) : (
          <div className={styles.treeContainer}>
            {tree?.map((node) => (
              <TreeNode key={node.name} node={node} selectedPath={selectedPath} onSelect={setSelectedPath} />
            ))}
            {tree?.length === 0 && <div className={styles.empty}>No documents yet</div>}
          </div>
        )}
      </div>

      {/* Right: Document Viewer */}
      <div className={styles.viewer}>
        {doc ? (
          <>
            <div className={styles.docHeader}>
              <div className={styles.docPath}>{doc.path.replace(/\/[^/]+$/, '/')}</div>
              <h2 className={styles.docTitle}>{doc.title}</h2>
              <div className={styles.docActions}>
                {mentions && mentions.length > 0 && (
                  <button className={styles.refsBtn} onClick={() => setShowRefs(!showRefs)}>
                    References ({mentions.length})
                  </button>
                )}
                <Link to={`/projects/${slug}/kb/edit/${doc.path}`} className={styles.editBtn}>Edit</Link>
                <button className={styles.deleteBtn} onClick={handleDelete}>Delete</button>
              </div>
            </div>
            {doc.tags && (
              <div className={styles.tagRow}>
                {doc.tags.split(',').map((t) => (
                  <span key={t.trim()} className={styles.tagChip} onClick={() => setActiveTag(t.trim())}>
                    {t.trim()}
                  </span>
                ))}
              </div>
            )}
            <div className={styles.docMeta}>
              Updated by <strong>{doc.updated_by}</strong> · {new Date(doc.updated_at).toLocaleString()}
            </div>
            {showRefs && mentions && (
              <div className={styles.referencesPanel}>
                <div className={styles.refsPanelTitle}>Referenced By</div>
                {mentions.map((m) => (
                  <div key={m.id} className={styles.refItem}>
                    <span className={styles.refSource}>{m.source_type}</span> #{m.source_id}
                  </div>
                ))}
              </div>
            )}
            <div className={styles.docContent}>
              <MentionRenderer text={doc.content} />
            </div>
          </>
        ) : (
          <div className={styles.placeholder}>
            Select a document from the tree, or <Link to={`/projects/${slug}/kb/new`}>create a new one</Link>.
          </div>
        )}
      </div>
    </div>
  );
}
