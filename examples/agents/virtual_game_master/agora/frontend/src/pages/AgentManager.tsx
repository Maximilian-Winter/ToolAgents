import { useState, type FormEvent, type ChangeEvent } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useAgents } from '../hooks/useAgents';
import { useProject } from '../hooks/useProjects';
import {
  useProjectAgents, useAddProjectAgent, useUpdateProjectAgent, useRemoveProjectAgent,
} from '../hooks/useProjectAgents';
import {
  useCustomFieldDefinitions, useAgentFields, useSetAgentFields,
} from '../hooks/useCustomFields';
import {
  Button, Input, TextArea, Select, Modal, FormField, Badge,
  EmptyState, Avatar,
} from '../components/ui';
import { cx } from '../lib/cx';
import type { ProjectAgent, CustomFieldDefinition } from '../api/types';
import styles from './AgentManager.module.css';

// ── Runtime badge colour helper ──────────────────────────────────────────────

function runtimeColor(runtime: string | null): string {
  if (!runtime) return 'var(--text-muted)';
  if (runtime === 'claude-code') return 'var(--accent-indigo, #6366f1)';
  if (runtime === 'aider') return 'var(--accent-green)';
  return 'var(--text-secondary)';
}

const RUNTIME_OPTIONS = [
  { value: '', label: 'Default' },
  { value: 'claude-code', label: 'claude-code' },
  { value: 'aider', label: 'aider' },
  { value: '__custom__', label: 'Custom…' },
];

// ── AgentCard ────────────────────────────────────────────────────────────────

function AgentCard({
  pa,
  fieldDefs,
  onEdit,
  onRemove,
  confirmRemove,
  setConfirmRemove,
  isRemoving,
}: {
  pa: ProjectAgent;
  fieldDefs: CustomFieldDefinition[];
  onEdit: (pa: ProjectAgent) => void;
  onRemove: (name: string) => void;
  confirmRemove: string | null;
  setConfirmRemove: (v: string | null) => void;
  isRemoving: boolean;
}) {
  const { data: agentFields } = useAgentFields(pa.agent_name);

  const chips = fieldDefs
    .filter((d) => agentFields?.[d.name] != null && agentFields[d.name] !== '')
    .map((d) => ({ label: d.label, value: agentFields![d.name] }));

  const isConfirming = confirmRemove === pa.agent_name;

  return (
    <div className={styles.agentCard}>
      {/* Header */}
      <div className={styles.cardHeader}>
        <Avatar name={pa.agent_name} size="md" />
        <div className={styles.cardTitle}>
          <span className={styles.displayName}>
            {pa.agent_display_name ?? pa.agent_name}
          </span>
          <span className={styles.agentHandle}>@{pa.agent_name}</span>
        </div>
        {pa.runtime && (
          <span
            className={styles.runtimeBadge}
            style={{ color: runtimeColor(pa.runtime), borderColor: runtimeColor(pa.runtime) }}
          >
            {pa.runtime}
          </span>
        )}
      </div>

      {/* Body */}
      <div className={styles.cardBody}>
        {pa.agent_role && (
          <p className={styles.roleText}>{pa.agent_role}</p>
        )}
        {chips.length > 0 && (
          <div className={styles.chipRow}>
            {chips.map((c) => (
              <span key={c.label} className={styles.chip}>
                <span className={styles.chipLabel}>{c.label}:</span> {c.value}
              </span>
            ))}
          </div>
        )}
        {pa.model && (
          <Badge variant="subtle" className={styles.modelBadge}>{pa.model}</Badge>
        )}
      </div>

      {/* Footer */}
      <div className={styles.cardFooter}>
        {isConfirming ? (
          <div className={styles.confirmRow}>
            <span className={styles.confirmText}>Remove agent?</span>
            <Button size="sm" variant="danger" onClick={() => onRemove(pa.agent_name)} loading={isRemoving}>
              Confirm
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirmRemove(null)}>Cancel</Button>
          </div>
        ) : (
          <>
            <Button size="sm" variant="ghost" onClick={() => onEdit(pa)}>Edit</Button>
            <Button size="sm" variant="danger" onClick={() => setConfirmRemove(pa.agent_name)}>Remove</Button>
          </>
        )}
      </div>
    </div>
  );
}

// ── AddAgentModal ─────────────────────────────────────────────────────────────

function AddAgentModal({
  open,
  onClose,
  availableAgents,
  onAdd,
  isPending,
  error,
}: {
  open: boolean;
  onClose: () => void;
  availableAgents: { name: string; role: string | null }[];
  onAdd: (agentName: string, runtime: string) => void;
  isPending: boolean;
  error: Error | null;
}) {
  const [selectedAgent, setSelectedAgent] = useState('');
  const [runtime, setRuntime] = useState('');
  const [customRuntime, setCustomRuntime] = useState('');

  const handleClose = () => {
    setSelectedAgent('');
    setRuntime('');
    setCustomRuntime('');
    onClose();
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!selectedAgent) return;
    const effectiveRuntime = runtime === '__custom__' ? customRuntime.trim() : runtime;
    onAdd(selectedAgent, effectiveRuntime);
  };

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Add Agent to Project"
      description="Select an agent from the global registry."
      footer={
        <div className={styles.formActions}>
          <Button variant="secondary" onClick={handleClose}>Cancel</Button>
          <Button
            onClick={handleSubmit}
            disabled={!selectedAgent}
            loading={isPending}
          >
            Add to Project
          </Button>
        </div>
      }
    >
      <form onSubmit={handleSubmit} className={styles.formFields}>
        <FormField label="Agent">
          {availableAgents.length === 0 ? (
            <div className={styles.emptyNote}>
              All agents already added or none exist.{' '}
              <Link to="/agents" className={styles.inlineLink}>
                Create one in the registry
              </Link>
            </div>
          ) : (
            <Select
              value={selectedAgent}
              onChange={(e: ChangeEvent<HTMLSelectElement>) => setSelectedAgent(e.target.value)}
              placeholder="Select an agent..."
              options={availableAgents.map((a) => ({
                value: a.name,
                label: `${a.name}${a.role ? ` — ${a.role}` : ''}`,
              }))}
            />
          )}
        </FormField>

        <FormField label="Runtime">
          <Select
            value={runtime}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => setRuntime(e.target.value)}
            options={RUNTIME_OPTIONS}
          />
          {runtime === '__custom__' && (
            <Input
              className={styles.customRuntimeInput}
              value={customRuntime}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setCustomRuntime(e.target.value)}
              placeholder="e.g. my-custom-runner"
              autoFocus
            />
          )}
        </FormField>

        {error && <div className={styles.error}>{(error as Error).message}</div>}
      </form>
    </Modal>
  );
}

// ── CustomFieldEditor ─────────────────────────────────────────────────────────

function CustomFieldEditor({
  defs,
  values,
  onChange,
}: {
  defs: CustomFieldDefinition[];
  values: Record<string, string>;
  onChange: (name: string, value: string) => void;
}) {
  if (defs.length === 0) {
    return <p className={styles.noCustomFields}>No custom field definitions yet.</p>;
  }

  return (
    <div className={styles.customFieldList}>
      {defs.map((def) => {
        const val = values[def.name] ?? def.default_value ?? '';

        if (def.field_type === 'boolean') {
          return (
            <FormField key={def.name} label={def.label}>
              <label className={styles.checkRow}>
                <input
                  type="checkbox"
                  checked={val === 'true'}
                  onChange={(e) => onChange(def.name, e.target.checked ? 'true' : 'false')}
                />
                <span>{def.label}</span>
              </label>
            </FormField>
          );
        }

        if (def.field_type === 'number') {
          return (
            <FormField key={def.name} label={def.label}>
              <Input
                type="number"
                value={val}
                onChange={(e: ChangeEvent<HTMLInputElement>) => onChange(def.name, e.target.value)}
                placeholder={def.default_value ?? ''}
              />
            </FormField>
          );
        }

        if (def.field_type === 'enum') {
          let opts: { value: string; label: string }[] = [];
          if (def.options_json) {
            try {
              const parsed = JSON.parse(def.options_json);
              if (Array.isArray(parsed)) {
                opts = parsed.map((o: string) => ({ value: o, label: o }));
              }
            } catch {
              // ignore
            }
          }
          return (
            <FormField key={def.name} label={def.label}>
              <Select
                value={val}
                onChange={(e: ChangeEvent<HTMLSelectElement>) => onChange(def.name, e.target.value)}
                options={opts}
                placeholder="Select…"
              />
            </FormField>
          );
        }

        // default: string
        return (
          <FormField key={def.name} label={def.label}>
            <Input
              value={val}
              onChange={(e: ChangeEvent<HTMLInputElement>) => onChange(def.name, e.target.value)}
              placeholder={def.default_value ?? ''}
            />
          </FormField>
        );
      })}
    </div>
  );
}

// ── EditAgentModal ────────────────────────────────────────────────────────────

function EditAgentModal({
  pa,
  onClose,
  fieldDefs,
  onSave,
  isPending,
}: {
  pa: ProjectAgent;
  onClose: () => void;
  fieldDefs: CustomFieldDefinition[];
  onSave: (
    pa: ProjectAgent,
    patch: {
      runtime?: string;
      model?: string;
      prompt_source?: string;
      allowed_tools?: string;
      system_prompt?: string;
      initial_task?: string;
      extra_flags?: string;
    },
    customFields: Record<string, string>,
  ) => void;
  isPending: boolean;
}) {
  const { data: existingFields } = useAgentFields(pa.agent_name);

  const [customRuntime, setCustomRuntime] = useState(
    pa.runtime && !RUNTIME_OPTIONS.some((o) => o.value === pa.runtime) ? pa.runtime : '',
  );
  const [runtimeMode, setRuntimeMode] = useState<string>(() => {
    if (!pa.runtime) return '';
    if (RUNTIME_OPTIONS.some((o) => o.value === pa.runtime && o.value !== '__custom__')) return pa.runtime;
    return '__custom__';
  });

  const [model, setModel] = useState(pa.model ?? '');
  const [promptSource, setPromptSource] = useState(pa.prompt_source ?? 'append');
  const [allowedTools, setAllowedTools] = useState(pa.allowed_tools ?? '');
  const [systemPrompt, setSystemPrompt] = useState(pa.system_prompt ?? '');
  const [initialTask, setInitialTask] = useState(pa.initial_task ?? '');
  const [extraFlags, setExtraFlags] = useState(pa.extra_flags ?? '');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [customFieldValues, setCustomFieldValues] = useState<Record<string, string>>(
    () => existingFields ?? {},
  );

  // Sync when existingFields loads
  const fieldsLoaded = existingFields !== undefined;
  if (fieldsLoaded && Object.keys(customFieldValues).length === 0 && Object.keys(existingFields!).length > 0) {
    setCustomFieldValues(existingFields!);
  }

  const handleRuntimeChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setRuntimeMode(e.target.value);
  };

  const handleCustomRuntimeChange = (e: ChangeEvent<HTMLInputElement>) => {
    setCustomRuntime(e.target.value);
  };

  const handleCustomFieldChange = (name: string, value: string) => {
    setCustomFieldValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSave = () => {
    const effectiveRuntime = runtimeMode === '__custom__' ? customRuntime.trim() : runtimeMode;
    onSave(
      pa,
      {
        runtime: effectiveRuntime || undefined,
        model: model || undefined,
        prompt_source: promptSource,
        allowed_tools: allowedTools || undefined,
        system_prompt: systemPrompt || undefined,
        initial_task: initialTask || undefined,
        extra_flags: extraFlags || undefined,
      },
      customFieldValues,
    );
  };

  return (
    <Modal
      open
      onClose={onClose}
      title={`Edit — ${pa.agent_display_name ?? pa.agent_name}`}
      description={`@${pa.agent_name}${pa.agent_role ? ` · ${pa.agent_role}` : ''}`}
      footer={
        <div className={styles.formActions}>
          <Button variant="secondary" onClick={onClose}>Cancel</Button>
          <Button onClick={handleSave} loading={isPending}>Save Changes</Button>
        </div>
      }
    >
      <div className={styles.editGrid}>
        {/* Left column: Core Settings */}
        <div className={styles.editLeft}>
          <h3 className={styles.columnHeading}>Core Settings</h3>

          <FormField label="Display Name" hint="From global agent (read-only)">
            <Input value={pa.agent_display_name ?? pa.agent_name} disabled style={{ opacity: 0.6 }} />
          </FormField>

          <FormField label="Role" hint="From global agent (read-only)">
            <Input value={pa.agent_role ?? ''} disabled style={{ opacity: 0.6 }} />
          </FormField>

          <FormField label="Runtime">
            <Select
              value={runtimeMode}
              onChange={handleRuntimeChange}
              options={RUNTIME_OPTIONS}
            />
            {runtimeMode === '__custom__' && (
              <Input
                className={styles.customRuntimeInput}
                value={customRuntime}
                onChange={handleCustomRuntimeChange}
                placeholder="e.g. my-custom-runner"
              />
            )}
          </FormField>

          <FormField label="Model">
            <Input
              value={model}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setModel(e.target.value)}
              placeholder="e.g. claude-sonnet-4-6"
            />
          </FormField>

          <FormField label="Prompt Source">
            <Select
              value={promptSource}
              onChange={(e: ChangeEvent<HTMLSelectElement>) => setPromptSource(e.target.value)}
              options={[
                { value: 'append', label: 'Append' },
                { value: 'override', label: 'Override' },
              ]}
            />
          </FormField>

          <FormField label="Allowed Tools" hint="Comma-separated">
            <Input
              value={allowedTools}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setAllowedTools(e.target.value)}
              placeholder="Bash,Read,Write,Edit,Glob,Grep"
            />
          </FormField>
        </div>

        {/* Right column: Custom Fields */}
        <div className={styles.editRight}>
          <h3 className={styles.columnHeading}>Custom Fields</h3>
          <CustomFieldEditor
            defs={fieldDefs}
            values={customFieldValues}
            onChange={handleCustomFieldChange}
          />
        </div>
      </div>

      {/* Bottom: prompts */}
      <div className={styles.editBottom}>
        <FormField label="System Prompt">
          <TextArea
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="Override or extend the system prompt for this agent in this project..."
            rows={4}
          />
        </FormField>

        <FormField label="Initial Task">
          <TextArea
            value={initialTask}
            onChange={(e) => setInitialTask(e.target.value)}
            placeholder="Initial task or message given to the agent on launch..."
            rows={3}
          />
        </FormField>

        {/* Advanced: extra_flags */}
        <div className={styles.advancedSection}>
          <button
            type="button"
            className={styles.advancedToggle}
            onClick={() => setShowAdvanced((v) => !v)}
          >
            <span className={cx(styles.advancedChevron, showAdvanced && styles.advancedChevronOpen)}>
              &#9654;
            </span>
            Advanced
          </button>

          {showAdvanced && (
            <FormField label="Extra Flags" hint="JSON object of additional CLI flags">
              <TextArea
                value={extraFlags}
                onChange={(e) => setExtraFlags(e.target.value)}
                placeholder='{"--no-auto-commits": true}'
                rows={3}
                style={{ fontFamily: 'monospace', fontSize: 12 }}
              />
            </FormField>
          )}
        </div>
      </div>
    </Modal>
  );
}

// ── AgentManager (main page) ──────────────────────────────────────────────────

export default function AgentManager() {
  const { slug } = useParams<{ slug: string }>();
  const { data: project } = useProject(slug);
  const { data: allAgents } = useAgents();
  const { data: projectAgents, isLoading } = useProjectAgents(slug);
  const { data: fieldDefs } = useCustomFieldDefinitions('agent');

  const addAgent = useAddProjectAgent(slug);
  const updatePA = useUpdateProjectAgent(slug);
  const removeAgent = useRemoveProjectAgent(slug);

  const [showAddModal, setShowAddModal] = useState(false);
  const [editingAgent, setEditingAgent] = useState<ProjectAgent | null>(null);
  const [confirmRemove, setConfirmRemove] = useState<string | null>(null);

  const addedNames = new Set(projectAgents?.map((pa) => pa.agent_name) ?? []);
  const availableAgents = allAgents?.filter((a) => !addedNames.has(a.name)) ?? [];
  const hasAgents = projectAgents && projectAgents.length > 0;

  // ── Handlers ──

  const handleAdd = (agentName: string, runtime: string) => {
    addAgent.mutate(
      { agent_name: agentName, runtime: runtime || undefined },
      {
        onSuccess: () => {
          setShowAddModal(false);
        },
      },
    );
  };

  const setAgentFields = useSetAgentFields(editingAgent?.agent_name ?? '');

  const handleSaveEdit = (
    pa: ProjectAgent,
    patch: {
      runtime?: string;
      model?: string;
      prompt_source?: string;
      allowed_tools?: string;
      system_prompt?: string;
      initial_task?: string;
      extra_flags?: string;
    },
    customFields: Record<string, string>,
  ) => {
    updatePA.mutate(
      { agent_name: pa.agent_name, ...patch },
      {
        onSuccess: () => {
          // Save custom fields too
          const nonEmpty = Object.fromEntries(
            Object.entries(customFields).filter(([, v]) => v !== ''),
          );
          if (Object.keys(nonEmpty).length > 0) {
            setAgentFields.mutate(nonEmpty);
          }
          setEditingAgent(null);
        },
      },
    );
  };

  const handleRemove = (name: string) => {
    removeAgent.mutate(name, {
      onSuccess: () => {
        setConfirmRemove(null);
        if (editingAgent?.agent_name === name) setEditingAgent(null);
      },
    });
  };

  // ─────────────────────────────────────────────────────────────────────────

  if (isLoading) return <EmptyState message="Loading agents..." />;

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.pageHeader}>
        <div>
          <h1 className={styles.pageTitle}>Team Agents</h1>
          {project && (
            <p className={styles.pageSubtitle}>{project.name}</p>
          )}
        </div>
        <div className={styles.headerActions}>
          <Link to="/agents" className={styles.registryLink}>Registry</Link>
          <Button onClick={() => setShowAddModal(true)}>+ Add Agent</Button>
        </div>
      </div>

      {/* Content */}
      {!hasAgents ? (
        <EmptyState
          message="No agents assigned to this project."
          action={
            <div className={styles.emptyActions}>
              <Button onClick={() => setShowAddModal(true)}>Add Agent</Button>
              <Link to="/agents" className={styles.registryLink}>
                or manage the registry
              </Link>
            </div>
          }
        />
      ) : (
        <>
          <p className={styles.agentCount}>
            {projectAgents.length} agent{projectAgents.length !== 1 ? 's' : ''}
          </p>

          <div className={styles.cardGrid}>
            {projectAgents.map((pa) => (
              <AgentCard
                key={pa.agent_name}
                pa={pa}
                fieldDefs={fieldDefs ?? []}
                onEdit={setEditingAgent}
                onRemove={handleRemove}
                confirmRemove={confirmRemove}
                setConfirmRemove={setConfirmRemove}
                isRemoving={removeAgent.isPending}
              />
            ))}
          </div>

          {/* Documents link */}
          <div className={styles.docsLink}>
            <Link to={`/projects/${slug}/documents`} className={styles.docsLinkAnchor}>
              Generate documents for this team →
            </Link>
          </div>
        </>
      )}

      {/* Add Agent Modal */}
      <AddAgentModal
        open={showAddModal}
        onClose={() => setShowAddModal(false)}
        availableAgents={availableAgents}
        onAdd={handleAdd}
        isPending={addAgent.isPending}
        error={addAgent.error as Error | null}
      />

      {/* Edit Agent Modal */}
      {editingAgent && (
        <EditAgentModal
          pa={editingAgent}
          onClose={() => setEditingAgent(null)}
          fieldDefs={fieldDefs ?? []}
          onSave={handleSaveEdit}
          isPending={updatePA.isPending || setAgentFields.isPending}
        />
      )}
    </div>
  );
}
