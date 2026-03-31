import { type ButtonHTMLAttributes } from 'react';
import { cx } from '../../lib/cx';
import styles from './IconButton.module.css';

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  icon: string;
  tooltip?: string;
  variant?: 'default' | 'danger';
}

export function IconButton({
  icon,
  tooltip,
  variant = 'default',
  className,
  ...rest
}: IconButtonProps) {
  return (
    <button
      className={cx(
        styles.iconButton,
        variant === 'danger' && styles.danger,
        className
      )}
      {...rest}
    >
      {icon}
      {tooltip && <span className={styles.tooltip}>{tooltip}</span>}
    </button>
  );
}
