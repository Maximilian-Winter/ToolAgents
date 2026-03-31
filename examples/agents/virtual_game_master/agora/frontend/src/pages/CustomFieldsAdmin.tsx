import { useState, type FormEvent } from 'react';
import {
  useCustomFieldDefinitions,
  useCreateFieldDefinition,
  useUpdateFieldDefinition,
  useDeleteFieldDefinition,
} from '../hooks/useCustomFields';
import {
  Button, Input, Modal, FormField, Select, Badge, EmptyState, Tabs,
} from '../components/ui';
import type { CustomFieldDefinition } from '../api/types';
import styles from './CustomFieldsAdmin.module.css';

type EntityType = 'agent' | 'project';

const FIELD_TYPE_OPTIONS = [
  { value: 'string', label: 'String' },
  { value: 'number', label: 'Number' },
  { value: 'boolean', label: 'Boolean' },
  { value: 'enum', label: 'Enum' },
];

const TABS = [
  { id: 'agent', label: 'Agent Fields' },
  { id: 'project', label: 'Project Fields' },
];

const FIELD_TYPE_COLORS: Record<string, string> = {
  string: 'var(--accent-blue)',
  number: 'var(--accent-green)',
  boolean: 'var(--accent-yellow)',
  enum: 'var(--accent-purple)',
};

interface FieldFormState {
  name: string;
  label: string;
  field_type: string;
  options: string; // comma-separated, for enum
  default_value: string;
  required: boolean;
}

const DEFAULT_FORM: FieldFormState = {
  name: '',
  label: '',
  field_type: 'string',
  options: '',
  default_value: '',
  required: false,
};

function fieldFromDefinition(def: CustomFieldDefinition): FieldFormState {
  let options = '';
  if (def.options_json) {
    try {
      const parsed = JSON.parse(def.options_json);
      if (Array.isArray(parsed)) options = parsed.join(', ');
    } catch {
      options = def.options_json;
    }
  }
  return {
    name: def.name,
    label: def.label,
    field_type: def.field_type,
    options,
    default_value: def.default_value ?? '',
    required: def.required,
  };
}

function optionsToJson(csv: string): string | undefined {
  const trimmed = csv.trim();
  if (!trimmed) return undefined;
  const arr = trimmed.split(',').map((s) => s.trim()).filter(Boolean);
  return JSON.stringify(arr);
}

function FieldRow({
  field,
  onEdit,
  onDelete,
  confirmDeleteId,
  setConfirmDeleteId,
  isDeleting,
}: {
  field: CustomFieldDefinition;
  onEdit: (f: CustomFieldDefinition) => void;
  onDelete: (id: number) => void;
  confirmDeleteId: number | null;
  setConfirmDeleteId: (id: number | null) => void;
  isDeleting: boolean;
}) {
  let optionsDisplay: string | null = null;
  if (field.options_json) {
    try {
      const parsed = JSON.parse(field.options_json);
      if (Array.isArray(parsed)) optionsDisplay = parsed.join(', ');
    } catch {
      optionsDisplay = field.options_json;
    }
  }

  return (
    <div className={styles.fieldRow}>
      <div className={styles.colName}>
        <span className={styles.fieldName}>{field.label}</span>
        <span className={styles.fieldMachine}>{field.name}</span>
      </div>

      <div className={styles.colType}>
        <Badge color={FIELD_TYPE_COLORS[field.field_type] ?? 'var(--text-muted)'}>
          {field.field_type}
        </Badge>
      </div>

      <div className={styles.colRequired}>
        {field.required ? (
          <Badge color="var(--accent-red)">Required</Badge>
        ) : (
          <span className={styles.optional}>—</span>
        )}
      </div>

      <div className={styles.colDefault}>
        {field.default_value ? (
          <code className={styles.defaultCode}>{field.default_value}</code>
        ) : optionsDisplay ? (
          <span className={styles.enumOptions}>{optionsDisplay}</span>
        ) : (
          <span className={styles.noDefault}>—</span>
        )}
      </div>

      <div className={styles.colActions}>
        {confirmDeleteId === field.id ? (
          <>
            <Button size="sm" variant="danger" onClick={() => onDelete(field.id)} loading={isDeleting}>
              Delete
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirmDeleteId(null)}>
              Cancel
            </Button>
          </>
        ) : (
          <>
            <Button size="sm" variant="ghost" onClick={() => onEdit(field)}>Edit</Button>
            <Button size="sm" variant="danger" onClick={() => setConfirmDeleteId(field.id)}>Delete</Button>
          </>
        )}
      </div>
    </div>
  );
}

export default function CustomFieldsAdmin() {
  const [activeTab, setActiveTab] = useState<EntityType>('agent');
  const { data: fields, isLoading } = useCustomFieldDefinitions(activeTab);
  const createField = useCreateFieldDefinition();
  const updateField = useUpdateFieldDefinition();
  const deleteField = useDeleteFieldDefinition();

  const [showModal, setShowModal] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [form, setForm] = useState<FieldFormState>(DEFAULT_FORM);
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);

  const openCreate = () => {
    setEditingId(null);
    setForm({ ...DEFAULT_FORM });
    setShowModal(true);
  };

  const openEdit = (f: CustomFieldDefinition) => {
    setEditingId(f.id);
    setForm(fieldFromDefinition(f));
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setEditingId(null);
    createField.reset?.();
    updateField.reset?.();
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!form.name.trim() || !form.label.trim()) return;

    const options_json =
      form.field_type === 'enum' ? optionsToJson(form.options) : undefined;

    if (editingId !== null) {
      updateField.mutate(
        {
          id: editingId,
          label: form.label,
          options_json,
          default_value: form.default_value || undefined,
          required: form.required,
        },
        { onSuccess: closeModal },
      );
    } else {
      createField.mutate(
        {
          name: form.name.trim(),
          label: form.label.trim(),
          field_type: form.field_type,
          entity_type: activeTab,
          options_json,
          default_value: form.default_value || undefined,
          required: form.required,
        },
        { onSuccess: closeModal },
      );
    }
  };

  const handleDelete = (id: number) => {
    deleteField.mutate(id, { onSuccess: () => setConfirmDeleteId(null) });
  };

  const updateForm = (patch: Partial<FieldFormState>) =>
    setForm((prev) => ({ ...prev, ...patch }));

  const isSubmitting = createField.isPending || updateField.isPending;
  const submitError = createField.error || updateField.error;

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Custom Fields</h1>
          <p className={styles.subtitle}>
            Manage custom field definitions for agents and projects.
          </p>
        </div>
        <Button onClick={openCreate}>+ Add Field</Button>
      </div>

      <Tabs
        tabs={TABS}
        active={activeTab}
        onChange={(id) => {
          setActiveTab(id as EntityType);
          setConfirmDeleteId(null);
        }}
        className={styles.tabs}
      />

      {isLoading ? (
        <EmptyState message="Loading fields..." />
      ) : (fields?.length ?? 0) === 0 ? (
        <EmptyState
          message={`No ${activeTab} fields defined yet.`}
          action={<Button onClick={openCreate}>Add Field</Button>}
        />
      ) : (
        <div className={styles.table}>
          <div className={styles.tableHeader}>
            <div className={styles.colName}>Name</div>
            <div className={styles.colType}>Type</div>
            <div className={styles.colRequired}>Required</div>
            <div className={styles.colDefault}>Default / Options</div>
            <div className={styles.colActions}>Actions</div>
          </div>
          {fields!.map((f) => (
            <FieldRow
              key={f.id}
              field={f}
              onEdit={openEdit}
              onDelete={handleDelete}
              confirmDeleteId={confirmDeleteId}
              setConfirmDeleteId={setConfirmDeleteId}
              isDeleting={deleteField.isPending}
            />
          ))}
        </div>
      )}

      <Modal
        open={showModal}
        onClose={closeModal}
        title={editingId !== null ? 'Edit Field' : 'Add Field'}
        description={
          editingId !== null
            ? 'Update the field definition.'
            : `Add a new custom field for ${activeTab}s.`
        }
        footer={
          <div className={styles.formActions}>
            <Button variant="secondary" onClick={closeModal}>
              Cancel
            </Button>
            <Button
              onClick={handleSubmit}
              disabled={!form.name.trim() || !form.label.trim()}
              loading={isSubmitting}
            >
              {editingId !== null ? 'Save Changes' : 'Add Field'}
            </Button>
          </div>
        }
      >
        <form onSubmit={handleSubmit} className={styles.formFields}>
          <FormField label="Name" hint="Lowercase letters, numbers, and underscores only">
            <Input
              value={form.name}
              onChange={(e) =>
                updateForm({ name: e.target.value.replace(/[^a-z0-9_]/g, '') })
              }
              placeholder="e.g. slack_handle"
              disabled={editingId !== null}
              autoFocus={editingId === null}
              style={editingId !== null ? { opacity: 0.6 } : undefined}
            />
          </FormField>

          <FormField label="Label" hint="Human-readable display name">
            <Input
              value={form.label}
              onChange={(e) => updateForm({ label: e.target.value })}
              placeholder="e.g. Slack Handle"
              autoFocus={editingId !== null}
            />
          </FormField>

          <FormField label="Field Type">
            <Select
              options={FIELD_TYPE_OPTIONS}
              value={form.field_type}
              onChange={(e) => updateForm({ field_type: e.target.value, options: '' })}
              disabled={editingId !== null}
            />
          </FormField>

          <FormField label="Entity Type">
            <Input
              value={activeTab}
              disabled
              style={{ opacity: 0.6 }}
            />
          </FormField>

          {form.field_type === 'enum' && (
            <FormField label="Options" hint="Comma-separated list of allowed values">
              <Input
                value={form.options}
                onChange={(e) => updateForm({ options: e.target.value })}
                placeholder="e.g. low, medium, high"
              />
            </FormField>
          )}

          <FormField label="Default Value" hint="Optional default value">
            <Input
              value={form.default_value}
              onChange={(e) => updateForm({ default_value: e.target.value })}
              placeholder="Leave blank for no default"
            />
          </FormField>

          <FormField label="Required">
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={form.required}
                onChange={(e) => updateForm({ required: e.target.checked })}
              />
              <span>This field is required</span>
            </label>
          </FormField>

          {submitError && (
            <div className={styles.error}>
              {(submitError as Error).message}
            </div>
          )}
        </form>
      </Modal>
    </div>
  );
}
