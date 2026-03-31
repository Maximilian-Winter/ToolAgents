import { type CSSProperties } from 'react';
import { cx } from '../../lib/cx';
import styles from './Avatar.module.css';

export interface AvatarProps {
  name: string;
  status?: 'online' | 'idle' | 'offline';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const COLORS = [
  '#7a8ef7', '#7ecf6a', '#e06070', '#e0a84c',
  '#b07af7', '#5cc9d0', '#f07890', '#60c0a0',
];

function hashName(name: string): number {
  let h = 0;
  for (let i = 0; i < name.length; i++) {
    h = ((h << 5) - h + name.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

export function Avatar({ name, status, size = 'md', className }: AvatarProps) {
  const color = COLORS[hashName(name) % COLORS.length];
  const initial = name.charAt(0);
  const style: CSSProperties = { background: color };

  return (
    <div className={cx(styles.wrapper, className)}>
      <div
        className={cx(styles.avatar, size === 'sm' && styles.sm, size === 'lg' && styles.lg)}
        style={style}
        title={name}
      >
        {initial}
      </div>
      {status && <span className={cx(styles.statusDot, styles[status])} />}
    </div>
  );
}
