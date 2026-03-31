import { useState, type FormEvent } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useIssue, useComments, useCreateComment, useUpdateIssue } from '../hooks/useTasks';
import {
  Button, Input, TextArea, Badge, Avatar, EmptyState,
} from '../components/ui';
import MentionRenderer from '../components/MentionRenderer';
import styles from './TaskDetail.module.css';

const PRIORITY_COLORS: Record<string, string> = {
  critical: 'var(--accent-red)',
  high: 'var(--accent-amber)',
  medium: 'var(--accent-blue)',
  low: 'var(--text-muted)',
};

export default function TaskDetail() {
  const { slug, number } = useParams<{ slug: string; number: string }>();
  const issueNumber = number ? parseInt(number, 10) : undefined;
  const { data: issue, isLoading } = useIssue(slug, issueNumber);
  const { data: comments } = useComments(slug, issueNumber);
  const createComment = useCreateComment(slug, issueNumber);
  const updateIssue = useUpdateIssue(slug, issueNumber);

  const [commentBody, setCommentBody] = useState('');
  const [author, setAuthor] = useState('user');

  const handleComment = (e: FormEvent) => {
    e.preventDefault();
    if (!commentBody.trim()) return;
    createComment.mutate(
      { author, body: commentBody.trim() },
      { onSuccess: () => setCommentBody('') }
    );
  };

  const toggleState = () => {
    if (!issue) return;
    updateIssue.mutate({ state: issue.state === 'open' ? 'closed' : 'open' });
  };

  if (isLoading) return <EmptyState message="Loading issue..." />;
  if (!issue) return <EmptyState message="Issue not found." />;

  const pColor = PRIORITY_COLORS[issue.priority] || 'var(--text-muted)';

  return (
    <div className={styles.page}>
      <div className={styles.main}>
        <Link to={`/projects/${slug}/issues`} className={styles.backLink}>
          &larr; Back to issues
        </Link>
        <h1 className={styles.title}>{issue.title}</h1>
        <div className={styles.issueNumber}>#{issue.number}</div>

        {issue.body && <div className={styles.body}><MentionRenderer text={issue.body} /></div>}

        <div className={styles.sectionTitle}>
          Comments ({comments?.length ?? 0})
        </div>

        {comments?.map((c) => (
          <div key={c.id} className={styles.comment}>
            <div className={styles.commentHeader}>
              <Avatar name={c.author} size="sm" />
              <span className={styles.commentAuthor}>{c.author}</span>
              <span className={styles.commentDate}>
                {new Date(c.created_at).toLocaleString()}
              </span>
            </div>
            <div className={styles.commentBody}><MentionRenderer text={c.body} /></div>
          </div>
        ))}

        {comments && comments.length === 0 && (
          <EmptyState message="No comments yet." />
        )}

        <form className={styles.commentForm} onSubmit={handleComment}>
          <div className={styles.commentFormRow}>
            <Input
              inputSize="sm"
              value={author}
              onChange={(e) => setAuthor(e.target.value)}
              placeholder="Author"
              style={{ width: 120 }}
            />
            <Button
              type="submit"
              size="sm"
              loading={createComment.isPending}
            >
              Comment
            </Button>
          </div>
          <TextArea
            value={commentBody}
            onChange={(e) => setCommentBody(e.target.value)}
            placeholder="Add a comment..."
            rows={3}
          />
        </form>
      </div>

      <div className={styles.sidebar}>
        <div className={styles.sidebarTitle}>Details</div>

        <div className={styles.metaRow}>
          <span className={styles.metaLabel}>State</span>
          <span
            className={styles.metaValue}
            style={{
              color: issue.state === 'open' ? 'var(--accent-green)' : 'var(--text-muted)',
            }}
          >
            {issue.state}
          </span>
        </div>
        <div className={styles.metaRow}>
          <span className={styles.metaLabel}>Priority</span>
          <Badge color={pColor} variant="subtle">{issue.priority}</Badge>
        </div>
        <div className={styles.metaRow}>
          <span className={styles.metaLabel}>Reporter</span>
          <span className={styles.metaValue}>{issue.reporter}</span>
        </div>
        <div className={styles.metaRow}>
          <span className={styles.metaLabel}>Assignee</span>
          <span className={styles.metaValue}>{issue.assignee || 'Unassigned'}</span>
        </div>
        <div className={styles.metaRow}>
          <span className={styles.metaLabel}>Created</span>
          <span className={styles.metaValue}>
            {new Date(issue.created_at).toLocaleDateString()}
          </span>
        </div>
        {issue.closed_at && (
          <div className={styles.metaRow}>
            <span className={styles.metaLabel}>Closed</span>
            <span className={styles.metaValue}>
              {new Date(issue.closed_at).toLocaleDateString()}
            </span>
          </div>
        )}

        {issue.labels && issue.labels.length > 0 && (
          <>
            <div className={styles.sectionTitle}>Labels</div>
            <div className={styles.labels}>
              {issue.labels.map((l) => (
                <Badge key={l.id} color={l.color || undefined} variant="subtle">
                  {l.name}
                </Badge>
              ))}
            </div>
          </>
        )}

        <button
          className={styles.stateToggle}
          style={{
            borderColor: issue.state === 'open' ? 'var(--accent-red)' : 'var(--accent-green)',
            color: issue.state === 'open' ? 'var(--accent-red)' : 'var(--accent-green)',
          }}
          onClick={toggleState}
          disabled={updateIssue.isPending}
        >
          {issue.state === 'open' ? 'Close Issue' : 'Reopen Issue'}
        </button>
      </div>
    </div>
  );
}
