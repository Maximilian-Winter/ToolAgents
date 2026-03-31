import { type TextareaHTMLAttributes, useRef, useCallback, useEffect } from 'react';
import { cx } from '../../lib/cx';
import styles from './TextArea.module.css';

export interface TextAreaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  autoResize?: boolean;
}

export function TextArea({
  autoResize = false,
  className,
  onChange,
  value,
  ...rest
}: TextAreaProps) {
  const ref = useRef<HTMLTextAreaElement>(null);

  const resize = useCallback(() => {
    if (autoResize && ref.current) {
      ref.current.style.height = 'auto';
      ref.current.style.height = ref.current.scrollHeight + 'px';
    }
  }, [autoResize]);

  useEffect(() => { resize(); }, [value, resize]);

  return (
    <textarea
      ref={ref}
      className={cx(styles.textarea, autoResize && styles.noResize, className)}
      value={value}
      onChange={(e) => {
        onChange?.(e);
        resize();
      }}
      {...rest}
    />
  );
}
