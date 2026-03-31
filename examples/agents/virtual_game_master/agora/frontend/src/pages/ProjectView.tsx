import { useParams, NavLink, Outlet, useLocation } from 'react-router-dom';
import { useProject } from '../hooks/useProjects';
import { cx } from '../lib/cx';
import Terminals from './Terminals';
import styles from './ProjectView.module.css';

const TABS = [
  { path: 'overview', label: 'Overview' },
  { path: 'chat', label: 'Chat' },
  { path: 'issues', label: 'Issues' },
  { path: 'documents', label: 'Documents' },
  { path: 'kb', label: 'Knowledge Base' },
  { path: 'agents', label: 'Agents' },
  { path: 'terminals', label: 'Terminals' },
  { path: 'settings', label: 'Settings' },
];

export default function ProjectView() {
  const { slug } = useParams<{ slug: string }>();
  const { data: project, isLoading } = useProject(slug);
  const location = useLocation();
  const isTerminals = location.pathname.endsWith('/terminals');

  if (isLoading) {
    return <div className={styles.loading}>Loading project...</div>;
  }

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <span className={styles.title}>{project?.name ?? slug}</span>
        {project?.description && (
          <span className={styles.desc}>{project.description}</span>
        )}
      </div>

      <div className={styles.tabs}>
        {TABS.map((tab) => (
          <NavLink
            key={tab.path}
            to={`/projects/${slug}/${tab.path}`}
            className={({ isActive }) =>
              cx(styles.tab, isActive && styles.tabActive)
            }
          >
            {tab.label}
          </NavLink>
        ))}
      </div>

      {/* Outlet for non-terminal routes */}
      <div className={styles.content} style={{ display: isTerminals ? 'none' : undefined }}>
        <Outlet />
      </div>
      {/* Terminals always mounted, hidden when not on terminals route */}
      <div className={styles.content} style={{ display: isTerminals ? undefined : 'none' }}>
        <Terminals />
      </div>
    </div>
  );
}
