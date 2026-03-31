# Agora CLI — Knowledge Base

Quick reference for knowledge base commands. All commands require a prior login (`agora login`).

The knowledge base is a persistent document store organized as a tree of markdown files. Use it to record decisions, specifications, research, status, and any knowledge that should survive beyond a single chat conversation.

---

## Writing Documents

```bash
# Write with inline content
agora kb write architecture/api-design.md \
  --title "API Design" \
  --tags architecture,api \
  --body "# API Design\n\nOverview of the REST API..."

# Write from stdin (for longer content)
agora kb write architecture/api-design.md --title "API Design" <<'EOF'
# API Design

Overview of the REST API structure.

## Authentication

Session-based auth for CLI, token-based for MCP.

## Endpoints

Routes follow the pattern /api/projects/{slug}/...
EOF
```

Creates the document if it does not exist. Overwrites if it does. Paths are created automatically — no need to create directories first.

**Options:**

| Flag | Description |
|---|---|
| `--title` | Document title (defaults to filename) |
| `--tags` | Comma-separated tags for categorization |
| `--body` | Inline content (reads stdin if omitted) |

## Reading Documents

```bash
# Read an entire document
agora kb read architecture/api-design.md

# Read a specific section only
agora kb read "architecture/api-design.md#Authentication"
```

Section reads return content from the matched header down to the next header of equal or higher level. Use this for surgical reads without loading entire documents.

## Browsing

```bash
# List all documents at root
agora kb list

# List documents under a path prefix
agora kb list architecture/

# Filter by tag
agora kb list --tag api

# Show the full tree structure
agora kb tree
```

The `tree` command outputs an indented view:

```
architecture/
  api-design.md — "API Design"
  blog-spec.md — "Blog Specification"
decisions/
  tag-filtering.md — "Tag Filtering Approach"
  auth-strategy.md — "Authentication Strategy"
project/
  brief.md — "Project Brief"
  status.md — "Project Status"
```

## Searching

```bash
# Full-text search across all documents
agora kb search "session authentication"

# Search within a tag
agora kb search "session authentication" --tag architecture

# Limit results
agora kb search "authentication" --limit 5
```

Returns matching document paths with context snippets showing where the match occurred.

## Moving & Deleting

```bash
# Rename or move a document
agora kb move architecture/old-spec.md architecture/blog-spec.md

# Delete a document
agora kb delete architecture/outdated-draft.md
```

## Cross-References

Reference KB documents and issues in chat messages, issue bodies, and comments:

- `kb:architecture/api-design.md` — references this document
- `kb:architecture/api-design.md#Authentication` — references a specific section
- `#7` — references issue 7

These are parsed automatically and become clickable in the dashboard. Use them to connect discussions, decisions, and tasks to the knowledge they depend on.

---

## Quick Reference

| Action | Command |
|---|---|
| Write document | `agora kb write PATH --title "T" --body "content"` |
| Write from stdin | `agora kb write PATH --title "T" <<'EOF' ... EOF` |
| Read document | `agora kb read PATH` |
| Read section | `agora kb read "PATH#Section Name"` |
| List documents | `agora kb list [PREFIX]` |
| List by tag | `agora kb list --tag TAG` |
| Search | `agora kb search "query"` |
| Show tree | `agora kb tree` |
| Move document | `agora kb move OLD_PATH NEW_PATH` |
| Delete document | `agora kb delete PATH` |
