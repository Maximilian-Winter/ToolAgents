import { type InputHTMLAttributes } from 'react';
import { cx } from '../../lib/cx';
import styles from './Input.module.css';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  error?: boolean;
  inputSize?: 'sm' | 'md';
}

export function Input({
  error,
  inputSize = 'md',
  className,
  ...rest
}: InputProps) {
  return (
    <input
      className={cx(
        styles.input,
        inputSize === 'sm' && styles.sm,
        error && styles.inputError,
        className
      )}
      {...rest}
    />
  );
}
