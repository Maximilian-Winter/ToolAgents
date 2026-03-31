import { type ReactNode } from 'react';
import { cx } from '../../lib/cx';
import styles from './Section.module.css';

export interface SectionProps {
  title?: string;
  description?: string;
  action?: ReactNode;
  variant?: 'default' | 'danger';
  className?: string;
  children: ReactNode;
}

export function Section({
  title,
  description,
  action,
  variant = 'default',
  className,
  children,
}: SectionProps) {
  return (
    <div className={cx(styles.section, variant === 'danger' && styles.danger, className)}>
      {title && (
        <div className={styles.header}>
          <div>
            <div className={cx(styles.title, variant === 'danger' && styles.dangerTitle)}>
              {title}
            </div>
            {description && <div className={styles.description}>{description}</div>}
          </div>
          {action}
        </div>
      )}
      <div className={styles.body}>{children}</div>
    </div>
  );
}
