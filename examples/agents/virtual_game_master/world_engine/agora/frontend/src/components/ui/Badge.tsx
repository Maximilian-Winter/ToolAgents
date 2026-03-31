import { type ReactNode, type CSSProperties } from 'react';
import { cx } from '../../lib/cx';
import styles from './Badge.module.css';

export interface BadgeProps {
  children: ReactNode;
  color?: string;
  variant?: 'solid' | 'subtle' | 'outline';
  dot?: boolean;
  className?: string;
}

export function Badge({
  children,
  color,
  variant = 'subtle',
  dot = false,
  className,
}: BadgeProps) {
  const style = color ? ({ '--color': color } as CSSProperties) : undefined;

  return (
    <span className={cx(styles.badge, styles[variant], className)} style={style}>
      {dot && <span className={styles.dot} style={style} />}
      {children}
    </span>
  );
}
