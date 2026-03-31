import { type SelectHTMLAttributes } from 'react';
import { cx } from '../../lib/cx';
import styles from './Select.module.css';

export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  options: SelectOption[];
  selectSize?: 'sm' | 'md';
  placeholder?: string;
}

export function Select({
  options,
  selectSize = 'md',
  placeholder,
  className,
  ...rest
}: SelectProps) {
  return (
    <select
      className={cx(styles.select, selectSize === 'sm' && styles.sm, className)}
      {...rest}
    >
      {placeholder && <option value="">{placeholder}</option>}
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}
