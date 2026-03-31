import { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { useKBDocument, useCreateOrReplaceKBDoc } from '../hooks/useKnowledgeBase';
import MentionRenderer from '../components/MentionRenderer';
import styles from './KBEditor.module.css';

export default function KBEditor() {
  const { slug, '*': splatPath } = useParams<{ slug: string; '*': string }>();
  const navigate = useNavigate();
  const location = useLocation();

  // Determine if editing (path from URL splat) or creating new
  const isEdit = location.pathname.includes('/kb/edit/');
  const editPath = isEdit ? splatPath : undefined;

  const { data: existingDoc } = useKBDocument(slug, editPath);
  const saveMut = useCreateOrReplaceKBDoc(slug);

  const [path, setPath] = useState('');
  const [title, setTitle] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState('');
  const [content, setContent] = useState('');
  const [mode, setMode] = useState<'write' | 'preview'>('write');

  // Pre-fill form when editing
  useEffect(() => {
    if (existingDoc) {
      setPath(existingDoc.path);
      setTitle(existingDoc.title);
      setTags(existingDoc.tags ? existingDoc.tags.split(',').map((t) => t.trim()) : []);
      setContent(existingDoc.content);
    }
  }, [existingDoc]);

  const addTag = () => {
    const t = tagInput.trim();
    if (t && !tags.includes(t)) {
      setTags([...tags, t]);
    }
    setTagInput('');
  };

  const removeTag = (tag: string) => {
    setTags(tags.filter((t) => t !== tag));
  };

  const handleTagKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      addTag();
    }
  };

  const handleSave = () => {
    if (!path.trim() || !slug) return;
    saveMut.mutate(
      {
        path: path.trim(),
        title: title.trim() || undefined,
        tags: tags.length > 0 ? tags.join(',') : undefined,
        content,
        author: 'admin', // TODO: get from auth context if available
      },
      {
        onSuccess: () => navigate(`/projects/${slug}/kb`),
      },
    );
  };

  const insertMarkdown = (before: string, after: string) => {
    const textarea = document.querySelector<HTMLTextAreaElement>(`.${styles.textarea}`);
    if (!textarea) return;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selected = content.slice(start, end);
    const newContent = content.slice(0, start) + before + selected + after + content.slice(end);
    setContent(newContent);
  };

  return (
    <div className={styles.form}>
      <h2>{isEdit ? 'Edit Document' : 'New Document'}</h2>

      <div className={styles.row}>
        <div className={styles.field}>
          <label className={styles.fieldLabel}>Path</label>
          <input
            className={styles.input}
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="architecture/api-design.md"
            disabled={isEdit}
          />
        </div>
        <div className={styles.field}>
          <label className={styles.fieldLabel}>Title</label>
          <input
            className={styles.input}
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Document title"
          />
        </div>
      </div>

      <div className={styles.field}>
        <label className={styles.fieldLabel}>Tags</label>
        <div className={styles.tags}>
          {tags.map((t) => (
            <span key={t} className={styles.tagChip}>
              {t} <span className={styles.tagRemove} onClick={() => removeTag(t)}>×</span>
            </span>
          ))}
          <input
            className={styles.tagInput}
            value={tagInput}
            onChange={(e) => setTagInput(e.target.value)}
            onKeyDown={handleTagKeyDown}
            onBlur={addTag}
            placeholder="Add tag..."
          />
        </div>
      </div>

      <div className={styles.editorSection}>
        <div className={styles.editorHeader}>
          <div className={styles.editorTabs}>
            <button className={`${styles.tab} ${mode === 'write' ? styles.tabActive : ''}`} onClick={() => setMode('write')}>Write</button>
            <button className={`${styles.tab} ${mode === 'preview' ? styles.tabActive : ''}`} onClick={() => setMode('preview')}>Preview</button>
          </div>
          {mode === 'write' && (
            <div className={styles.toolbar}>
              <button className={styles.toolBtn} onClick={() => insertMarkdown('**', '**')} title="Bold"><b>B</b></button>
              <button className={styles.toolBtn} onClick={() => insertMarkdown('*', '*')} title="Italic"><i>I</i></button>
              <button className={styles.toolBtn} onClick={() => insertMarkdown('`', '`')} title="Code">&lt;/&gt;</button>
              <button className={styles.toolBtn} onClick={() => insertMarkdown('[', '](url)')} title="Link">&#x1f517;</button>
              <button className={styles.toolBtn} onClick={() => insertMarkdown('## ', '')} title="Heading">H</button>
            </div>
          )}
        </div>
        {mode === 'write' ? (
          <textarea
            className={styles.textarea}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Write markdown content..."
          />
        ) : (
          <div className={styles.preview}>
            <MentionRenderer text={content} />
          </div>
        )}
      </div>

      <div className={styles.actions}>
        <button className={styles.cancelBtn} onClick={() => navigate(`/projects/${slug}/kb`)}>Cancel</button>
        <button className={styles.saveBtn} onClick={handleSave} disabled={!path.trim() || saveMut.isPending}>
          {saveMut.isPending ? 'Saving...' : 'Save Document'}
        </button>
      </div>
    </div>
  );
}
