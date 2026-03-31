import { useState, type FormEvent } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  useProjectTemplates,
  useCreateProjectTemplate,
  useUpdateTemplate,
  useDeleteTemplate,
  useRenderTemplate,
} from '../hooks/useTemplates';
import { useProjectAgents } from '../hooks/useProjectAgents';
import {
  Button, Modal, FormField, Input, TextArea, Badge, EmptyState, Select,
} from '../components/ui';
import { downloadFile, saveFilesToDisk, hasFileSystemAccess } from '../lib/fileUtils';
import type { DocumentTemplate } from '../api/types';
import styles from './DocumentsPage.module.css';

const CONTENT_PLACEHOLDER = `# {{agent.name}} — {{project.slug}}

{{agent.role}}

Project: {{project.name}}
{{project.description}}
`;

interface TemplateFormState {
  name: string;
  description: string;
  type_tag: string;
  content: string;
}

const DEFAULT_FORM: TemplateFormState = {
  name: '',
  description: '',
  type_tag: '',
  content: '',
};

function formFromTemplate(t: DocumentTemplate): TemplateFormState {
  return {
    name: t.name,
    description: t.description ?? '',
    type_tag: t.type_tag ?? '',
    content: t.content,
  };
}

function scopeOf(t: DocumentTemplate): 'global' | 'project' {
  return t.project_id === null ? 'global' : 'project';
}

function ScopeBadge({ scope }: { scope: 'global' | 'project' }) {
  return scope === 'global' ? (
    <Badge color="var(--accent-blue)">global</Badge>
  ) : (
    <Badge color="var(--accent-green)">project</Badge>
  );
}

function TemplateRow({
  template,
  onEdit,
  onDelete,
  onGenerate,
  confirmDeleteId,
  setConfirmDeleteId,
  isDeleting,
}: {
  template: DocumentTemplate;
  onEdit: (t: DocumentTemplate) => void;
  onDelete: (id: number) => void;
  onGenerate: (t: DocumentTemplate) => void;
  confirmDeleteId: number | null;
  setConfirmDeleteId: (id: number | null) => void;
  isDeleting: boolean;
}) {
  const scope = scopeOf(template);

  return (
    <div className={styles.templateRow}>
      <div className={styles.colName}>
        <span className={styles.templateName}>{template.name}</span>
        {template.description && (
          <span className={styles.templateDesc}>{template.description}</span>
        )}
      </div>

      <div className={styles.colTag}>
        {template.type_tag ? (
          <Badge color="var(--accent-purple)">{template.type_tag}</Badge>
        ) : (
          <span className={styles.noTag}>—</span>
        )}
      </div>

      <div className={styles.colScope}>
        <ScopeBadge scope={scope} />
      </div>

      <div className={styles.colActions}>
        <Button size="sm" variant="ghost" onClick={() => onGenerate(template)}>
          Generate
        </Button>
        {confirmDeleteId === template.id ? (
          <>
            <Button size="sm" variant="danger" onClick={() => onDelete(template.id)} loading={isDeleting}>
              Delete
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirmDeleteId(null)}>
              Cancel
            </Button>
          </>
        ) : (
          <>
            {scope === 'global' ? (
              <Link to="/templates" className={styles.editLink}>
                Edit in Templates
              </Link>
            ) : (
              <Button size="sm" variant="ghost" onClick={() => onEdit(template)}>Edit</Button>
            )}
            {scope === 'project' && (
              <Button size="sm" variant="danger" onClick={() => setConfirmDeleteId(template.id)}>
                Delete
              </Button>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default function DocumentsPage() {
  const { slug = '' } = useParams<{ slug: string }>();

  const { data: templates, isLoading } = useProjectTemplates(slug);
  const { data: projectAgents } = useProjectAgents(slug);
  const createTemplate = useCreateProjectTemplate(slug);
  const updateTemplate = useUpdateTemplate();
  const deleteTemplate = useDeleteTemplate();
  const renderTemplate = useRenderTemplate();

  // Template form modal
  const [showFormModal, setShowFormModal] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [form, setForm] = useState<TemplateFormState>(DEFAULT_FORM);
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);

  // Generate modal
  const [generateTarget, setGenerateTarget] = useState<DocumentTemplate | null>(null);
  const [selectedAgent, setSelectedAgent] = useState('');
  const [renderedContent, setRenderedContent] = useState('');
  const [unresolvedVars, setUnresolvedVars] = useState<string[]>([]);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  // Filter
  const [activeTag, setActiveTag] = useState<string>('All');

  const allTags = Array.from(
    new Set((templates ?? []).map((t) => t.type_tag).filter(Boolean) as string[])
  ).sort();

  const filtered = (templates ?? []).filter((t) => {
    if (activeTag === 'All') return true;
    return t.type_tag === activeTag;
  });

  // ─── Form modal ───
  const openCreate = () => {
    setEditingId(null);
    setForm({ ...DEFAULT_FORM });
    setShowFormModal(true);
  };

  const openEdit = (t: DocumentTemplate) => {
    setEditingId(t.id);
    setForm(formFromTemplate(t));
    setShowFormModal(true);
  };

  const closeFormModal = () => {
    setShowFormModal(false);
    setEditingId(null);
    createTemplate.reset?.();
    updateTemplate.reset?.();
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!form.name.trim() || !form.content.trim()) return;

    const payload = {
      name: form.name.trim(),
      description: form.description.trim() || undefined,
      type_tag: form.type_tag.trim() || undefined,
      content: form.content,
    };

    if (editingId !== null) {
      updateTemplate.mutate(
        { id: editingId, ...payload },
        { onSuccess: closeFormModal },
      );
    } else {
      createTemplate.mutate(payload, { onSuccess: closeFormModal });
    }
  };

  const handleDelete = (id: number) => {
    deleteTemplate.mutate(id, { onSuccess: () => setConfirmDeleteId(null) });
  };

  const updateForm = (patch: Partial<TemplateFormState>) =>
    setForm((prev) => ({ ...prev, ...patch }));

  const isSubmitting = createTemplate.isPending || updateTemplate.isPending;
  const submitError = createTemplate.error || updateTemplate.error;

  // ─── Generate modal ───
  const openGenerate = (t: DocumentTemplate) => {
    setGenerateTarget(t);
    setSelectedAgent('');
    setRenderedContent('');
    setUnresolvedVars([]);
    setSaveStatus(null);
    renderTemplate.reset();
  };

  const closeGenerateModal = () => {
    setGenerateTarget(null);
    renderTemplate.reset();
    setSaveStatus(null);
  };

  const handleGenerate = () => {
    if (!generateTarget) return;
    renderTemplate.mutate(
      {
        templateId: generateTarget.id,
        projectSlug: slug,
        agentName: selectedAgent || undefined,
      },
      {
        onSuccess: (data) => {
          setRenderedContent(data.rendered_content);
          setUnresolvedVars(data.unresolved_variables ?? []);
          setSaveStatus(null);
        },
      },
    );
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(renderedContent).then(() => {
      setSaveStatus('Copied to clipboard!');
      setTimeout(() => setSaveStatus(null), 2000);
    });
  };

  const handleDownload = () => {
    const filename = `${generateTarget?.name ?? 'template'}.md`;
    downloadFile(filename, renderedContent);
    setSaveStatus('Downloaded.');
    setTimeout(() => setSaveStatus(null), 2000);
  };

  const handleSaveToDisk = async () => {
    const filename = `${generateTarget?.name ?? 'template'}.md`;
    const count = await saveFilesToDisk([{ name: filename, content: renderedContent }]);
    if (count !== null) {
      setSaveStatus(`Saved ${count} file(s) to disk.`);
    } else {
      setSaveStatus('Save cancelled.');
    }
    setTimeout(() => setSaveStatus(null), 3000);
  };

  const agentOptions = [
    { value: '', label: 'No agent (project context only)' },
    ...(projectAgents ?? []).map((pa) => ({
      value: pa.agent_name,
      label: pa.agent_display_name ?? pa.agent_name,
    })),
  ];

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Documents</h1>
          <p className={styles.subtitle}>
            Generate documents from global and project-specific templates.
          </p>
        </div>
        <Button onClick={openCreate}>+ New Template</Button>
      </div>

      {/* Type tag filter chips */}
      {allTags.length > 0 && (
        <div className={styles.filterChips}>
          {(['All', ...allTags] as string[]).map((tag) => (
            <button
              key={tag}
              className={`${styles.chip} ${activeTag === tag ? styles.chipActive : ''}`}
              onClick={() => setActiveTag(tag)}
            >
              {tag}
            </button>
          ))}
        </div>
      )}

      {isLoading ? (
        <EmptyState message="Loading templates..." />
      ) : filtered.length === 0 ? (
        <EmptyState
          message={
            activeTag !== 'All'
              ? `No templates with type "${activeTag}".`
              : 'No templates for this project yet.'
          }
          action={
            activeTag === 'All' ? <Button onClick={openCreate}>New Template</Button> : undefined
          }
        />
      ) : (
        <div className={styles.table}>
          <div className={styles.tableHeader}>
            <div className={styles.colName}>Name / Description</div>
            <div className={styles.colTag}>Type</div>
            <div className={styles.colScope}>Scope</div>
            <div className={styles.colActions}>Actions</div>
          </div>
          {filtered.map((t) => (
            <TemplateRow
              key={t.id}
              template={t}
              onEdit={openEdit}
              onDelete={handleDelete}
              onGenerate={openGenerate}
              confirmDeleteId={confirmDeleteId}
              setConfirmDeleteId={setConfirmDeleteId}
              isDeleting={deleteTemplate.isPending}
            />
          ))}
        </div>
      )}

      {/* ─── Create / Edit Template Modal ─── */}
      <Modal
        open={showFormModal}
        onClose={closeFormModal}
        title={editingId !== null ? 'Edit Template' : 'New Project Template'}
        description={
          editingId !== null
            ? 'Update the template definition.'
            : 'Create a new template scoped to this project.'
        }
        footer={
          <div className={styles.formActions}>
            <Button variant="secondary" onClick={closeFormModal}>Cancel</Button>
            <Button
              onClick={handleSubmit}
              disabled={!form.name.trim() || !form.content.trim()}
              loading={isSubmitting}
            >
              {editingId !== null ? 'Save Changes' : 'Create Template'}
            </Button>
          </div>
        }
      >
        <form onSubmit={handleSubmit} className={styles.formFields}>
          <FormField label="Name">
            <Input
              value={form.name}
              onChange={(e) => updateForm({ name: e.target.value })}
              placeholder="e.g. Sprint Retrospective"
              autoFocus={editingId === null}
            />
          </FormField>

          <FormField label="Type Tag" hint="Category label (e.g. brief, report, spec)">
            <Input
              value={form.type_tag}
              onChange={(e) => updateForm({ type_tag: e.target.value })}
              placeholder="e.g. report"
            />
          </FormField>

          <FormField label="Description" hint="Optional summary">
            <Input
              value={form.description}
              onChange={(e) => updateForm({ description: e.target.value })}
              placeholder="e.g. End-of-sprint retrospective document"
            />
          </FormField>

          <FormField
            label="Content"
            hint="Use {{agent.name}}, {{agent.role}}, {{project.slug}}, {{project.name}}, {{project.description}}"
          >
            <TextArea
              value={form.content}
              onChange={(e) => updateForm({ content: e.target.value })}
              placeholder={CONTENT_PLACEHOLDER}
              rows={12}
              style={{ fontFamily: "'SF Mono', 'Fira Code', monospace", fontSize: '12px' }}
            />
          </FormField>

          {submitError && (
            <div className={styles.error}>{(submitError as Error).message}</div>
          )}
        </form>
      </Modal>

      {/* ─── Generate Modal ─── */}
      <Modal
        open={!!generateTarget}
        onClose={closeGenerateModal}
        title={`Generate: ${generateTarget?.name ?? ''}`}
        description="Select an optional agent context, then generate the document."
        footer={
          <div className={styles.formActions}>
            <Button variant="secondary" onClick={closeGenerateModal}>Close</Button>
            {renderedContent && (
              <>
                <Button variant="ghost" onClick={handleCopy}>Copy</Button>
                <Button variant="ghost" onClick={handleDownload}>Download</Button>
                {hasFileSystemAccess() && (
                  <Button variant="ghost" onClick={handleSaveToDisk}>Save to Disk</Button>
                )}
              </>
            )}
            <Button
              onClick={handleGenerate}
              loading={renderTemplate.isPending}
            >
              Generate
            </Button>
          </div>
        }
      >
        <div className={styles.generateForm}>
          <FormField label="Agent" hint="Optional — provides agent context variables">
            <Select
              options={agentOptions}
              value={selectedAgent}
              onChange={(e) => setSelectedAgent(e.target.value)}
            />
          </FormField>

          {unresolvedVars.length > 0 && (
            <div className={styles.warningBlock}>
              <span className={styles.warningLabel}>Unresolved variables:</span>
              <div className={styles.warningBadges}>
                {unresolvedVars.map((v) => (
                  <Badge key={v} color="var(--accent-yellow)">{v}</Badge>
                ))}
              </div>
            </div>
          )}

          {renderedContent && (
            <div className={styles.previewBlock}>
              <div className={styles.previewLabel}>Preview</div>
              <pre className={styles.preview}>{renderedContent}</pre>
            </div>
          )}

          {saveStatus && (
            <div className={styles.saveStatus}>{saveStatus}</div>
          )}

          {renderTemplate.error && (
            <div className={styles.error}>
              {(renderTemplate.error as Error).message}
            </div>
          )}
        </div>
      </Modal>
    </div>
  );
}
