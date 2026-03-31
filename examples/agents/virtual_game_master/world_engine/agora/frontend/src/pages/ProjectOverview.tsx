import { useParams, Link } from 'react-router-dom';
import { useProject } from '../hooks/useProjects';
import { useProjectStats } from '../hooks/useProjectStats';
import { useProjectAgents } from '../hooks/useProjectAgents';
import { Section, Badge, Avatar, EmptyState } from '../components/ui';
import styles from './ProjectOverview.module.css';

export default function ProjectOverview() {
  const { slug } = useParams<{ slug: string }>();
  const { data: project } = useProject(slug);
  const { data: stats } = useProjectStats(slug);
  const { data: projectAgents } = useProjectAgents(slug);

  return (
    <div className={styles.page}>
      {/* Stats */}
      <div className={styles.sectionGap}>
        <Section title="Dashboard">
          <div className={styles.statGrid}>
            <div className={styles.statCard}>
              <div className={styles.statValue}>{stats?.room_count ?? '\u2014'}</div>
              <div className={styles.statLabel}>Chat Rooms</div>
            </div>
            <div className={styles.statCard}>
              <div className={styles.statValue}>{stats?.open_issue_count ?? '\u2014'}</div>
              <div className={styles.statLabel}>Open Issues</div>
            </div>
            <div className={styles.statCard}>
              <div className={styles.statValue}>{projectAgents?.length ?? '\u2014'}</div>
              <div className={styles.statLabel}>Agents</div>
            </div>
          </div>
        </Section>
      </div>

      {/* Project Info */}
      <div className={styles.sectionGap}>
        <Section title="Project Info">
          <div className={styles.infoRow}>
            <span className={styles.infoLabel}>Slug</span>
            <span className={styles.infoValue}>{slug}</span>
          </div>
          {project?.description && (
            <div className={styles.infoRow}>
              <span className={styles.infoLabel}>Description</span>
              <span style={{ color: 'var(--text-primary)', fontSize: 13 }}>{project.description}</span>
            </div>
          )}
          {project?.working_dir && (
            <div className={styles.infoRow}>
              <span className={styles.infoLabel}>Working Dir</span>
              <span className={styles.infoValue}>{project.working_dir}</span>
            </div>
          )}
          <div className={styles.infoRow}>
            <span className={styles.infoLabel}>Created</span>
            <span style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
              {project ? new Date(project.created_at).toLocaleDateString() : '\u2014'}
            </span>
          </div>
        </Section>
      </div>

      {/* Agent Roster */}
      <div className={styles.sectionGap}>
        <Section
          title="Agent Roster"
          action={
            <Link to={`/projects/${slug}/agents`} style={{ fontSize: 12, color: 'var(--accent-blue)' }}>
              Manage Agents &rarr;
            </Link>
          }
        >
          {(!projectAgents || projectAgents.length === 0) ? (
            <EmptyState
              icon="🤖"
              message="No agents assigned yet."
              action={<Link to={`/projects/${slug}/agents`} style={{ color: 'var(--accent-blue)', fontSize: 13 }}>Add agents</Link>}
            />
          ) : (
            projectAgents.map((pa) => (
              <div key={pa.agent_name} className={styles.agentRow}>
                <Avatar name={pa.agent_name} size="sm" />
                <span className={styles.agentName}>{pa.agent_name}</span>
                <span className={styles.agentMeta}>
                  {pa.agent_display_name && <>{pa.agent_display_name} &middot; </>}
                  {pa.agent_role ?? 'No role'}
                </span>
                {pa.model && <Badge variant="subtle">{pa.model}</Badge>}
              </div>
            ))
          )}
        </Section>
      </div>

      {/* Quick Links */}
      <div className={styles.sectionGap}>
        <Section title="Quick Links">
          <div className={styles.quickLinks}>
            <Link to={`/projects/${slug}/chat`} className={styles.quickLink}>
              💬 Chat Rooms
            </Link>
            <Link to={`/projects/${slug}/issues`} className={styles.quickLink}>
              🐛 Issue Tracker
            </Link>
            <Link to={`/projects/${slug}/agents`} className={styles.quickLink}>
              🤖 Agent Config
            </Link>
            <Link to={`/projects/${slug}/settings`} className={styles.quickLink}>
              ⚙ Project Settings
            </Link>
          </div>
        </Section>
      </div>
    </div>
  );
}
