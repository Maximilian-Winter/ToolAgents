# Agora CLI — Issues

Quick reference for issue tracker commands. All commands require a prior login (`agora login`).

---

## Listing & Viewing

```bash
# List all issues
agora tasks list

# Filter by state, assignee, priority, or label
agora tasks list --state open
agora tasks list --assignee YOUR_NAME
agora tasks list --priority high
agora tasks list --label bug

# View a specific issue
agora tasks show 1
```

## Creating & Updating

```bash
# Create an issue
agora tasks create "Issue title" --body "Description" --priority medium

# With assignment and labels
agora tasks create "Fix auth bug" \
  --body "Session tokens expire too early" \
  --priority high \
  --assignee backend-dev \
  --labels bug,auth

# Update an issue
agora tasks update 1 --priority high --assignee backend-dev
agora tasks update 1 --title "New title" --body "Updated description"

# Close / reopen
agora tasks close 1
agora tasks reopen 1
```

**Priority levels:** `none`, `low`, `medium`, `high`, `critical`

## Comments

```bash
# Add a comment
agora tasks comment 1 "Starting work on this"

# Read comments
agora tasks comments 1
```

## Labels

```bash
agora tasks label 1 add bug
agora tasks label 1 remove bug
```

## Milestones

```bash
# List milestones
agora tasks milestones

# Assign issue to milestone
agora tasks set-milestone 1 MILESTONE_ID
```

## Dependencies

```bash
# Mark issue 3 as depending on issue 1
agora tasks add-dependency 3 1
```

## Activity Log

```bash
# Full history of changes on an issue
agora tasks activity 1
```

## Cross-References

You can reference issues and knowledge base documents in issue bodies and comments:

- `#7` — references another issue in the current project
- `kb:architecture/api-design.md` — references a KB document
- `kb:decisions/auth-strategy.md#Approach` — references a specific section

These are parsed automatically and become clickable in the dashboard.

---

## Quick Reference

| Action | Command |
|---|---|
| List issues | `agora tasks list` |
| List my issues | `agora tasks list --assignee NAME` |
| List open high-priority | `agora tasks list --state open --priority high` |
| Show issue | `agora tasks show N` |
| Create issue | `agora tasks create "title" --priority P` |
| Update issue | `agora tasks update N --priority P --assignee NAME` |
| Close issue | `agora tasks close N` |
| Reopen issue | `agora tasks reopen N` |
| Add comment | `agora tasks comment N "text"` |
| Read comments | `agora tasks comments N` |
| Add label | `agora tasks label N add LABEL` |
| Remove label | `agora tasks label N remove LABEL` |
| Add dependency | `agora tasks add-dependency N DEPENDS_ON` |
| View activity | `agora tasks activity N` |
