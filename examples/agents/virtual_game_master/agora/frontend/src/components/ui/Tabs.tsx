import { cx } from '../../lib/cx';
import styles from './Tabs.module.css';

export interface Tab {
  id: string;
  label: string;
}

export interface TabsProps {
  tabs: Tab[];
  active: string;
  onChange: (id: string) => void;
  className?: string;
}

export function Tabs({ tabs, active, onChange, className }: TabsProps) {
  return (
    <div className={cx(styles.tabList, className)}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={cx(styles.tab, tab.id === active && styles.active)}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
