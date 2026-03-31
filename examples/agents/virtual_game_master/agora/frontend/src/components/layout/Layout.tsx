import { NavLink, Outlet, useParams } from 'react-router-dom';
import { cx } from '../../lib/cx';
import styles from './Layout.module.css';

interface NavItemDef {
  to: string;
  icon: string;
  label: string;
  end?: boolean;
}

function NavItem({ to, icon, label, end }: NavItemDef) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cx(styles.navItem, isActive && styles.navItemActive)
      }
    >
      {icon}
      <span className={styles.navLabel}>{label}</span>
    </NavLink>
  );
}

export default function Layout() {
  const { slug } = useParams<{ slug: string }>();

  return (
    <div className={styles.shell}>
      <aside className={styles.activityBar}>
        {/* Brand / Home */}
        <NavLink to="/" className={styles.brand}>
          AT
        </NavLink>

        <div className={styles.divider} />

        {/* Global navigation — always visible */}
        <nav className={styles.nav}>
          <NavItem to="/" icon="📊" label="Dashboard" end />
          <NavItem to="/agents" icon="🤖" label="Agents" />
          <NavItem to="/custom-fields" icon="🗂" label="Custom Fields" />
          <NavItem to="/templates" icon="📄" label="Templates" />

          {/* Project navigation — only when inside a project */}
          {slug && (
            <>
              <div className={styles.projectDivider}>
                <span className={styles.projectLabel}>{slug}</span>
              </div>
              <NavItem to={`/projects/${slug}/overview`} icon="📋" label="Overview" />
              <NavItem to={`/projects/${slug}/chat`} icon="💬" label="Chat" />
              <NavItem to={`/projects/${slug}/issues`} icon="🐛" label="Issues" />
              <NavItem to={`/projects/${slug}/documents`} icon="📄" label="Documents" />
              <NavItem to={`/projects/${slug}/kb`} icon="📚" label="Knowledge Base" />
              <NavItem to={`/projects/${slug}/agents`} icon="⚙" label="Config" />
              <NavItem to={`/projects/${slug}/terminals`} icon="🖥" label="Terminals" />
            </>
          )}
        </nav>

        {/* Bottom items */}
        <div className={styles.bottomNav}>
          {slug && (
            <NavItem to={`/projects/${slug}/settings`} icon="⚙" label="Settings" />
          )}
        </div>
      </aside>

      <main className={styles.content}>
        <Outlet />
      </main>
    </div>
  );
}
