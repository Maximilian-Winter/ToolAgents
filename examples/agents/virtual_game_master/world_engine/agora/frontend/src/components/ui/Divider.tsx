import styles from './Divider.module.css';

export interface DividerProps {
  label?: string;
}

export function Divider({ label }: DividerProps) {
  if (!label) return <div className={styles.line} />;

  return (
    <div className={styles.divider}>
      <div className={styles.line} />
      <span>{label}</span>
      <div className={styles.line} />
    </div>
  );
}
