import { type ReactNode } from 'react';
import styles from './EmptyState.module.css';

export interface EmptyStateProps {
  icon?: string;
  message: string;
  action?: ReactNode;
}

export function EmptyState({ icon, message, action }: EmptyStateProps) {
  return (
    <div className={styles.empty}>
      {icon && <div className={styles.icon}>{icon}</div>}
      <div>{message}</div>
      {action}
    </div>
  );
}
