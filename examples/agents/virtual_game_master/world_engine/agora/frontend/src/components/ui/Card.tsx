import { type HTMLAttributes } from 'react';
import { cx } from '../../lib/cx';
import styles from './Card.module.css';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  interactive?: boolean;
}

export function Card({ interactive, className, children, ...rest }: CardProps) {
  return (
    <div
      className={cx(styles.card, interactive && styles.interactive, className)}
      {...rest}
    >
      {children}
    </div>
  );
}
