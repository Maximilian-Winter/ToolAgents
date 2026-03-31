import { useState, type FormEvent } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useIssues, useCreateIssue } from '../hooks/useTasks';
import {
  Button, Input, TextArea, Select, Modal, FormField, Badge, EmptyState,
} from '../components/ui';
import type { Issue } from '../api/types';
import styles from './TaskBoard.module.css';

const PRIORITIES = ['low', 'medium', 'high', 'critical'];

const PRIORITY_COLORS: Record<string, string> = {
  critical: 'var(--accent-red)',
  high: 'var(--accent-amber)',
  medium: 'var(--accent-blue)',
  low: 'var(--text-muted)',
};

const STATE_ICONS: Record<string, string> = {
  open: '\u25CB',
  closed: '\u2714',
};

function IssueRow({ issue, slug }: { issue: Issue; slug: string }) {
  const pColor = PRIORITY_COLORS[issue.priority] || 'var(--text-muted)';
  return (
    <Link to={`/projects/${slug}/issues/${issue.number}`} className={styles.issueRow}>
      <span
        className={styles.stateIcon}
        style={{ color: issue.state === 'open' ? 'var(--accent-green)' : 'var(--text-muted)' }}
      >
        {STATE_ICONS[issue.state] || '\u25CB'}
      </span>
      <span className={styles.issueNumber}>#{issue.number}</span>
      <span className={styles.issueTitle}>{issue.title}</span>
      {issue.labels?.map((l) => (
        <Badge key={l.id} color={l.color || undefined} variant="subtle">
          {l.name}
        </Badge>
      ))}
      <Badge color={pColor} variant="subtle">
        {issue.priority}
      </Badge>
      {issue.assignee && <span className={styles.assignee}>{issue.assignee}</span>}
      {issue.comment_count > 0 && (
        <span className={styles.commentCount}>{issue.comment_count} cmt</span>
      )}
    </Link>
  );
}

export default function TaskBoard() {
  const { slug } = useParams<{ slug: string }>();
  const [stateFilter, setStateFilter] = useState('');
  const [priorityFilter, setPriorityFilter] = useState('');
  const [assigneeFilter, setAssigneeFilter] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [title, setTitle] = useState('');
  const [body, setBody] = useState('');
  const [priority, setPriority] = useState('medium');
  const [reporter, setReporter] = useState('user');

  const filters: { state?: string; priority?: string; assignee?: string } = {};
  if (stateFilter) filters.state = stateFilter;
  if (priorityFilter) filters.priority = priorityFilter;
  if (assigneeFilter) filters.assignee = assigneeFilter;

  const { data: issues, isLoading } = useIssues(slug, filters);
  const createIssue = useCreateIssue(slug);

  const handleCreate = (e: FormEvent) => {
    e.preventDefault();
    if (!title.trim()) return;
    createIssue.mutate(
      { title: title.trim(), body: body.trim() || undefined, priority, reporter },
      {
        onSuccess: () => {
          setShowCreate(false);
          setTitle('');
          setBody('');
        },
      }
    );
  };

  return (
    <div className={styles.page}>
      <div className={styles.filterBar}>
        <Select
          selectSize="sm"
          value={stateFilter}
          onChange={(e) => setStateFilter(e.target.value)}
          placeholder="All states"
          options={[
            { value: 'open', label: 'Open' },
            { value: 'closed', label: 'Closed' },
          ]}
        />
        <Select
          selectSize="sm"
          value={priorityFilter}
          onChange={(e) => setPriorityFilter(e.target.value)}
          placeholder="All priorities"
          options={PRIORITIES.map((p) => ({ value: p, label: p }))}
        />
        <Input
          inputSize="sm"
          value={assigneeFilter}
          onChange={(e) => setAssigneeFilter(e.target.value)}
          placeholder="Assignee..."
          style={{ width: 120 }}
        />
        <div className={styles.filterRight}>
          <Button size="sm" onClick={() => setShowCreate(true)}>
            + New Issue
          </Button>
        </div>
      </div>

      <div className={styles.list}>
        {isLoading && <EmptyState message="Loading issues..." />}
        {issues && issues.length === 0 && (
          <EmptyState icon="🐛" message="No issues found." />
        )}
        {issues?.map((issue) => (
          <IssueRow key={issue.id} issue={issue} slug={slug!} />
        ))}
      </div>

      <Modal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        title="New Issue"
        footer={
          <div className={styles.formActions}>
            <Button variant="secondary" onClick={() => setShowCreate(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreate} loading={createIssue.isPending}>
              Create Issue
            </Button>
          </div>
        }
      >
        <form onSubmit={handleCreate} className={styles.formFields}>
          <FormField label="Title">
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Issue title"
              autoFocus
            />
          </FormField>
          <FormField label="Description">
            <TextArea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              placeholder="Describe the issue..."
              rows={4}
            />
          </FormField>
          <div className={styles.formRow}>
            <FormField label="Priority">
              <Select
                value={priority}
                onChange={(e) => setPriority(e.target.value)}
                options={PRIORITIES.map((p) => ({ value: p, label: p }))}
              />
            </FormField>
            <FormField label="Reporter">
              <Input
                value={reporter}
                onChange={(e) => setReporter(e.target.value)}
              />
            </FormField>
          </div>
        </form>
      </Modal>
    </div>
  );
}
