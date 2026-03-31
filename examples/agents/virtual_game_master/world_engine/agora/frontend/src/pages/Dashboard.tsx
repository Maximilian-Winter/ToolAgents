import { useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { useProjects, useCreateProject } from '../hooks/useProjects';
import { useAgents } from '../hooks/useAgents';
import { useProjectStats } from '../hooks/useProjectStats';
import { useProjectAgents } from '../hooks/useProjectAgents';
import { Button, Input, TextArea, Modal, FormField, EmptyState, Avatar } from '../components/ui';
import type { Project } from '../api/types';
import styles from './Dashboard.module.css';

/* ─── Per-project stats row inside the card ─── */
function ProjectCard({ project }: { project: Project }) {
  const { data: stats } = useProjectStats(project.slug);
  const { data: projectAgents } = useProjectAgents(project.slug);

  const agentCount = projectAgents?.length ?? 0;
  const roomCount = stats?.room_count ?? 0;
  const openIssues = stats?.open_issue_count ?? 0;
  const totalIssues = stats?.total_issue_count ?? 0;

  return (
    <Link to={`/projects/${project.slug}`} className={styles.cardLink}>
      <div className={styles.card}>
        {/* Card header */}
        <div className={styles.cardHeader}>
          <div className={styles.cardIcon}>
            {project.name.charAt(0).toUpperCase()}
          </div>
          <div className={styles.cardHeaderText}>
            <div className={styles.cardName}>{project.name}</div>
            <div className={styles.cardSlug}>{project.slug}</div>
          </div>
        </div>

        {/* Description */}
        {project.description && (
          <div className={styles.cardDesc}>{project.description}</div>
        )}

        {/* Stats grid */}
        <div className={styles.cardStatsGrid}>
          <div className={styles.cardStatItem}>
            <span className={styles.cardStatNum}>{agentCount}</span>
            <span className={styles.cardStatLabel}>Agents</span>
          </div>
          <div className={styles.cardStatItem}>
            <span className={styles.cardStatNum}>{roomCount}</span>
            <span className={styles.cardStatLabel}>Rooms</span>
          </div>
          <div className={styles.cardStatItem}>
            <span className={styles.cardStatNum}>{openIssues}</span>
            <span className={styles.cardStatLabel}>Open</span>
          </div>
          <div className={styles.cardStatItem}>
            <span className={styles.cardStatNum}>{totalIssues}</span>
            <span className={styles.cardStatLabel}>Issues</span>
          </div>
        </div>

        {/* Agent avatars */}
        {projectAgents && projectAgents.length > 0 && (
          <div className={styles.cardAgents}>
            {projectAgents.slice(0, 5).map((pa) => (
              <Avatar key={pa.agent_name} name={pa.agent_name} size="sm" />
            ))}
            {projectAgents.length > 5 && (
              <span className={styles.moreAgents}>+{projectAgents.length - 5}</span>
            )}
          </div>
        )}

        {/* Footer */}
        <div className={styles.cardFooter}>
          <span className={styles.cardDate}>
            Created {new Date(project.created_at).toLocaleDateString()}
          </span>
          {project.working_dir && (
            <span className={styles.cardDir} title={project.working_dir}>
              {project.working_dir.split(/[/\\]/).pop()}
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}

export default function Dashboard() {
  const { data: projects, isLoading, error } = useProjects();
  const { data: agents } = useAgents();
  const createProject = useCreateProject();
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const handleCreate = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    createProject.mutate(
      { name: name.trim(), description: description.trim() || undefined },
      {
        onSuccess: () => {
          setShowForm(false);
          setName('');
          setDescription('');
        },
      },
    );
  };

  if (isLoading) {
    return <div className={styles.page}><EmptyState message="Loading..." /></div>;
  }

  if (error) {
    return (
      <div className={styles.page}>
        <div className={styles.error}>Error: {(error as Error).message}</div>
      </div>
    );
  }

  const projectCount = projects?.length ?? 0;
  const agentCount = agents?.length ?? 0;

  return (
    <div className={styles.page}>
      {/* Page header */}
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Dashboard</h1>
          <p className={styles.subtitle}>Agora</p>
        </div>
        <Button onClick={() => setShowForm(true)}>+ New Project</Button>
      </div>

      {/* Global stats */}
      <div className={styles.globalStats}>
        <div className={styles.globalStatCard}>
          <span className={styles.globalStatValue}>{projectCount}</span>
          <span className={styles.globalStatLabel}>Projects</span>
        </div>
        <Link to="/agents" className={styles.globalStatCard} style={{ textDecoration: 'none' }}>
          <span className={styles.globalStatValue}>{agentCount}</span>
          <span className={styles.globalStatLabel}>Agent Templates</span>
        </Link>
      </div>

      {/* Projects */}
      <div className={styles.sectionHeader}>
        <h2 className={styles.sectionTitle}>Projects</h2>
        <span className={styles.sectionCount}>{projectCount}</span>
      </div>

      {projectCount === 0 && (
        <EmptyState
          icon="📁"
          message="No projects yet. Create one to get started."
          action={<Button onClick={() => setShowForm(true)}>Create Project</Button>}
        />
      )}

      <div className={styles.grid}>
        {projects?.map((p) => (
          <ProjectCard key={p.id} project={p} />
        ))}
      </div>

      {/* Quick links */}
      {projectCount > 0 && (
        <div className={styles.quickSection}>
          <h3 className={styles.quickTitle}>Quick Access</h3>
          <div className={styles.quickLinks}>
            <Link to="/agents" className={styles.quickLink}>
              🤖 Agent Registry ({agentCount} templates)
            </Link>
            {projects?.slice(0, 3).map((p) => (
              <Link key={p.slug} to={`/projects/${p.slug}/chat`} className={styles.quickLink}>
                💬 {p.name} Chat
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Create modal */}
      <Modal
        open={showForm}
        onClose={() => setShowForm(false)}
        title="New Project"
        footer={
          <div className={styles.formActions}>
            <Button variant="secondary" onClick={() => setShowForm(false)}>Cancel</Button>
            <Button onClick={handleCreate} loading={createProject.isPending}>Create</Button>
          </div>
        }
      >
        <form onSubmit={handleCreate} className={styles.formFields}>
          <FormField label="Name">
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Project"
              autoFocus
            />
          </FormField>
          <FormField label="Description" hint="Optional">
            <TextArea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What is this project about?"
              rows={3}
            />
          </FormField>
          {createProject.error && (
            <div className={styles.error}>{(createProject.error as Error).message}</div>
          )}
        </form>
      </Modal>
    </div>
  );
}
