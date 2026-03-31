import { useState, type FormEvent } from 'react';
import {
  useGlobalTemplates,
  useCreateGlobalTemplate,
  useUpdateTemplate,
  useDeleteTemplate,
} from '../hooks/useTemplates';
import {
  Button, Modal, FormField, Input, TextArea, Badge, EmptyState,
} from '../components/ui';
import type { DocumentTemplate } from '../api/types';
import styles from './TemplatesPage.module.css';

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

function TemplateRow({
  template,
  onEdit,
  onDelete,
  confirmDeleteId,
  setConfirmDeleteId,
  isDeleting,
}: {
  template: DocumentTemplate;
  onEdit: (t: DocumentTemplate) => void;
  onDelete: (id: number) => void;
  confirmDeleteId: number | null;
  setConfirmDeleteId: (id: number | null) => void;
  isDeleting: boolean;
}) {
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

      <div className={styles.colActions}>
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
            <Button size="sm" variant="ghost" onClick={() => onEdit(template)}>Edit</Button>
            <Button size="sm" variant="danger" onClick={() => setConfirmDeleteId(template.id)}>Delete</Button>
          </>
        )}
      </div>
    </div>
  );
}

export default function TemplatesPage() {
  const { data: templates, isLoading } = useGlobalTemplates();
  const createTemplate = useCreateGlobalTemplate();
  const updateTemplate = useUpdateTemplate();
  const deleteTemplate = useDeleteTemplate();

  const [showModal, setShowModal] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [form, setForm] = useState<TemplateFormState>(DEFAULT_FORM);
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);
  const [activeTag, setActiveTag] = useState<string>('All');

  // Dynamically extract unique type_tags
  const allTags = Array.from(
    new Set((templates ?? []).map((t) => t.type_tag).filter(Boolean) as string[])
  ).sort();

  const filtered = (templates ?? []).filter((t) => {
    if (activeTag === 'All') return true;
    return t.type_tag === activeTag;
  });

  const openCreate = () => {
    setEditingId(null);
    setForm({ ...DEFAULT_FORM });
    setShowModal(true);
  };

  const openEdit = (t: DocumentTemplate) => {
    setEditingId(t.id);
    setForm(formFromTemplate(t));
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
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
        { onSuccess: closeModal },
      );
    } else {
      createTemplate.mutate(payload, { onSuccess: closeModal });
    }
  };

  const handleDelete = (id: number) => {
    deleteTemplate.mutate(id, { onSuccess: () => setConfirmDeleteId(null) });
  };

  const updateForm = (patch: Partial<TemplateFormState>) =>
    setForm((prev) => ({ ...prev, ...patch }));

  const isSubmitting = createTemplate.isPending || updateTemplate.isPending;
  const submitError = createTemplate.error || updateTemplate.error;

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Templates</h1>
          <p className={styles.subtitle}>
            Manage global document templates available across all projects.
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
              : 'No global templates yet.'
          }
          action={activeTag === 'All' ? <Button onClick={openCreate}>New Template</Button> : undefined}
        />
      ) : (
        <div className={styles.table}>
          <div className={styles.tableHeader}>
            <div className={styles.colName}>Name / Description</div>
            <div className={styles.colTag}>Type</div>
            <div className={styles.colActions}>Actions</div>
          </div>
          {filtered.map((t) => (
            <TemplateRow
              key={t.id}
              template={t}
              onEdit={openEdit}
              onDelete={handleDelete}
              confirmDeleteId={confirmDeleteId}
              setConfirmDeleteId={setConfirmDeleteId}
              isDeleting={deleteTemplate.isPending}
            />
          ))}
        </div>
      )}

      <Modal
        open={showModal}
        onClose={closeModal}
        title={editingId !== null ? 'Edit Template' : 'New Template'}
        description={
          editingId !== null
            ? 'Update the template definition.'
            : 'Create a new global document template.'
        }
        footer={
          <div className={styles.formActions}>
            <Button variant="secondary" onClick={closeModal}>Cancel</Button>
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
              placeholder="e.g. Agent Onboarding Brief"
              autoFocus={editingId === null}
            />
          </FormField>

          <FormField label="Type Tag" hint="Category label (e.g. brief, report, spec)">
            <Input
              value={form.type_tag}
              onChange={(e) => updateForm({ type_tag: e.target.value })}
              placeholder="e.g. brief"
            />
          </FormField>

          <FormField label="Description" hint="Optional summary of what this template generates">
            <Input
              value={form.description}
              onChange={(e) => updateForm({ description: e.target.value })}
              placeholder="e.g. Generates an onboarding document for a new agent"
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
    </div>
  );
}
