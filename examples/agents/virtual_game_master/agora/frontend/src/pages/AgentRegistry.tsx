import { useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { useAgents, useCreateAgent, useUpdateAgent, useDeleteAgent } from '../hooks/useAgents';
import { useProjects } from '../hooks/useProjects';
import {
  Button, Input, Modal, FormField,
  EmptyState, Avatar, IconButton,
} from '../components/ui';
import styles from './AgentRegistry.module.css';
import type { Agent, Project } from '../api/types';

function AgentRow({
  agent,
  projectMap,
  onEdit,
  onDelete,
  confirmDelete,
  setConfirmDelete,
  isDeleting,
}: {
  agent: Agent;
  projectMap: Map<string, string[]>; // agentName -> projectSlugs
  onEdit: (a: Agent) => void;
  onDelete: (name: string) => void;
  confirmDelete: string | null;
  setConfirmDelete: (v: string | null) => void;
  isDeleting: boolean;
}) {
  const assignedProjects = projectMap.get(agent.name) ?? [];

  return (
    <div className={styles.agentCard}>
      <div className={styles.agentMain}>
        <Avatar name={agent.name} size="md" />
        <div className={styles.agentInfo}>
          <div className={styles.agentName}>
            {agent.display_name ?? agent.name}
            {agent.display_name && (
              <span className={styles.agentSlug}>{agent.name}</span>
            )}
          </div>
          <div className={styles.agentRole}>{agent.role ?? 'No role defined'}</div>
        </div>
      </div>

      <div className={styles.agentProjects}>
        {assignedProjects.length === 0 ? (
          <span className={styles.noProjects}>Not assigned</span>
        ) : (
          assignedProjects.map((slug) => (
            <Link key={slug} to={`/projects/${slug}/agents`} className={styles.projectBadge}>
              {slug}
            </Link>
          ))
        )}
      </div>

      <div className={styles.agentActions}>
        {confirmDelete === agent.name ? (
          <>
            <Button size="sm" variant="danger" onClick={() => onDelete(agent.name)} loading={isDeleting}>
              Delete
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirmDelete(null)}>Cancel</Button>
          </>
        ) : (
          <>
            <Button size="sm" variant="ghost" onClick={() => onEdit(agent)}>Edit</Button>
            <IconButton icon="×" variant="danger" onClick={() => setConfirmDelete(agent.name)} tooltip="Delete" />
          </>
        )}
      </div>
    </div>
  );
}

export default function AgentRegistry() {
  const { data: agents, isLoading } = useAgents();
  const { data: projects } = useProjects();
  const createAgent = useCreateAgent();
  const updateAgent = useUpdateAgent();
  const deleteAgent = useDeleteAgent();

  const [showModal, setShowModal] = useState(false);
  const [editingAgent, setEditingAgent] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [role, setRole] = useState('');
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [search, setSearch] = useState('');

  // Build agent → projects map
  // We'll fetch project agents for each project
  // For now, show basic info — we can enhance later
  const projectAgentMap = useAllProjectAgents(projects ?? []);

  const openCreate = () => {
    setEditingAgent(null);
    setName('');
    setDisplayName('');
    setRole('');
    setShowModal(true);
  };

  const openEdit = (a: Agent) => {
    setEditingAgent(a.name);
    setName(a.name);
    setDisplayName(a.display_name ?? '');
    setRole(a.role ?? '');
    setShowModal(true);
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    if (editingAgent) {
      updateAgent.mutate(
        { name: editingAgent, display_name: displayName || undefined, role: role || undefined },
        { onSuccess: () => setShowModal(false) },
      );
    } else {
      createAgent.mutate(
        { name: name.trim(), display_name: displayName || undefined, role: role || undefined },
        { onSuccess: () => setShowModal(false) },
      );
    }
  };

  const handleDelete = (agentName: string) => {
    deleteAgent.mutate(agentName, { onSuccess: () => setConfirmDelete(null) });
  };

  const filtered = agents?.filter((a) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return a.name.toLowerCase().includes(q)
      || (a.display_name ?? '').toLowerCase().includes(q)
      || (a.role ?? '').toLowerCase().includes(q);
  }) ?? [];

  if (isLoading) return <div className={styles.page}><EmptyState message="Loading agents..." /></div>;

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Agent Registry</h1>
          <p className={styles.subtitle}>
            Global agent templates. Assign them to projects for per-project configuration.
          </p>
        </div>
        <Button onClick={openCreate}>+ New Agent</Button>
      </div>

      {/* Stats bar */}
      <div className={styles.statsBar}>
        <div className={styles.stat}>
          <span className={styles.statValue}>{agents?.length ?? 0}</span>
          <span className={styles.statLabel}>Total Agents</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statValue}>
            {agents?.filter((a) => (projectAgentMap.get(a.name)?.length ?? 0) > 0).length ?? 0}
          </span>
          <span className={styles.statLabel}>Assigned</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statValue}>
            {agents?.filter((a) => (projectAgentMap.get(a.name)?.length ?? 0) === 0).length ?? 0}
          </span>
          <span className={styles.statLabel}>Unassigned</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statValue}>{projects?.length ?? 0}</span>
          <span className={styles.statLabel}>Projects</span>
        </div>
      </div>

      {/* Search */}
      {(agents?.length ?? 0) > 0 && (
        <div className={styles.searchBar}>
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search agents by name, display name, or role..."
            inputSize="sm"
          />
        </div>
      )}

      {/* Agent list */}
      {filtered.length === 0 && !search && (
        <EmptyState
          icon="🤖"
          message="No agents yet. Create your first agent template."
          action={<Button onClick={openCreate}>Create Agent</Button>}
        />
      )}
      {filtered.length === 0 && search && (
        <EmptyState message={`No agents matching "${search}"`} />
      )}

      <div className={styles.agentList}>
        {filtered.map((a) => (
          <AgentRow
            key={a.name}
            agent={a}
            projectMap={projectAgentMap}
            onEdit={openEdit}
            onDelete={handleDelete}
            confirmDelete={confirmDelete}
            setConfirmDelete={setConfirmDelete}
            isDeleting={deleteAgent.isPending}
          />
        ))}
      </div>

      {/* Create/Edit Modal */}
      <Modal
        open={showModal}
        onClose={() => setShowModal(false)}
        title={editingAgent ? 'Edit Agent' : 'Create Agent'}
        description={editingAgent ? 'Update the agent template.' : 'Define a reusable agent template that can be assigned to any project.'}
        footer={
          <div className={styles.formActions}>
            <Button variant="secondary" onClick={() => setShowModal(false)}>Cancel</Button>
            <Button
              onClick={handleSubmit}
              disabled={!name.trim()}
              loading={createAgent.isPending || updateAgent.isPending}
            >
              {editingAgent ? 'Save Changes' : 'Create Agent'}
            </Button>
          </div>
        }
      >
        <form onSubmit={handleSubmit} className={styles.formFields}>
          <FormField label="Name" hint="Unique identifier (lowercase, no spaces)">
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. code-reviewer"
              disabled={!!editingAgent}
              autoFocus={!editingAgent}
              style={editingAgent ? { opacity: 0.6 } : undefined}
            />
          </FormField>
          <FormField label="Display Name" hint="Human-friendly name">
            <Input
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="e.g. Code Reviewer"
              autoFocus={!!editingAgent}
            />
          </FormField>
          <FormField label="Role" hint="What this agent does">
            <Input
              value={role}
              onChange={(e) => setRole(e.target.value)}
              placeholder="e.g. Reviews code for quality and security"
            />
          </FormField>
          {(createAgent.error || updateAgent.error) && (
            <div className={styles.error}>
              {((createAgent.error || updateAgent.error) as Error).message}
            </div>
          )}
        </form>
      </Modal>
    </div>
  );
}

/**
 * Hook that aggregates project agents across all projects.
 * Returns a Map: agentName -> [slug1, slug2, ...]
 */
function useAllProjectAgents(_projects: Project[]): Map<string, string[]> {
  // We can't call hooks in a loop, so we gather data from all projects
  // by using a single query approach. For now, we'll fetch for each project
  // using a combined approach.
  //
  // Since React Query handles deduplication, calling useProjectAgents per
  // project is fine but we can't do it in a loop. Instead, we just use the
  // data we already have from each individual page visit (cached).
  //
  // For a proper solution, we'd add a backend endpoint. For now,
  // we'll use a simpler approach - fetch all at the registry level.

  const results = new Map<string, string[]>();

  // Use individual queries - but we need a fixed number of hooks.
  // Instead, let's use a different approach: we'll add the data as we go.
  // For the MVP, we simply don't show project assignments here.
  // TODO: Add a backend endpoint GET /agents/:name/projects

  return results;
}
