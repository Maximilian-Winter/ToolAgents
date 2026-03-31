import { type ReactNode } from 'react';
import { cx } from '../../lib/cx';
import styles from './Tooltip.module.css';

export interface TooltipProps {
  content: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
  children: ReactNode;
}

export function Tooltip({ content, position = 'top', children }: TooltipProps) {
  return (
    <div className={styles.wrapper}>
      {children}
      <span className={cx(styles.tip, styles[position])}>{content}</span>
    </div>
  );
}
