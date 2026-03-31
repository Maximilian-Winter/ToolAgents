# Agora — Consolidated Agent Feedback

All issues, feature requests, and suggestions from the Daggerfall report test, compiled from the editor, systems, and lore agent feedback documents.

Agents: editor, systems, lore
Date: 2026-03-15

---

## Bugs & Broken Behavior

| # | Issue | Reported By | Details |
|---|-------|-------------|---------|
| B1 | Unicode crash on Windows | editor, lore | `typer.echo()` crashes on non-ASCII characters (arrows, em dashes, subscripts) with `UnicodeEncodeError: 'charmap'`. Completely blocks `chat poll` and `kb read`. Fix: force UTF-8 in CLI entrypoint. |
| B2 | `chat wait` timeout throws full traceback | editor | 50-line Python traceback on normal timeout. Should catch exception and print clean message. |
| B3 | Empty documents created silently | editor, lore, systems | When stdin is empty or `cat` fails, `kb write` creates a zero-byte document with no warning. Indistinguishable from real content in `kb tree`. |
| B4 | `tasks update --state closed` rejected | systems | API rejects `--state closed` via update; must use separate `close` command. Unintuitive. |

---

## CLI Ergonomics — Writing

| # | Issue | Reported By | Details |
|---|-------|-------------|---------|
| W1 | `kb write --body` breaks with long content | all three | Heredocs break on apostrophes. Windows has 8191-char command limit. Shell quoting is fragile for prose. |
| W2 | No `--file` flag on `kb write` | all three | Most requested feature. `agora kb write PATH --file ./draft.md` avoids all quoting issues. |
| W3 | Heredoc example in guide breaks on apostrophes | lore, editor | `<<'EOF'` fails when content contains `'`. Guide should warn or suggest alternatives. |
| W4 | No confirmation of content size on write | systems, lore | `kb write` prints "Created" but not how much content was received. Should print line/byte count. |

---

## CLI Ergonomics — Reading & Orientation

| # | Issue | Reported By | Details |
|---|-------|-------------|---------|
| R1 | `kb tree` shows no metadata | editor | No size, timestamp, or author. Editor cannot distinguish placeholders from real drafts. |
| R2 | `kb read` on empty doc produces zero output | lore | Should print "(document is empty)" instead of nothing. |
| R3 | No `kb read --head` or `--summary` | editor | Editor wants to preview documents without loading full content. |
| R4 | `chat poll` requires room name | systems | No-argument `chat poll` errors instead of polling all rooms or listing available rooms. |
| R5 | No unified status command | editor, systems | Must poll chat, tasks, and KB separately to know what changed. Want single `agora status` or `agora catch-up`. |
| R6 | Issue state markers `O`/`X` unclear | editor, lore | Not immediately legible. Consider text labels or documentation. |

---

## Communication Model

| # | Issue | Reported By | Details |
|---|-------|-------------|---------|
| C1 | Chat rooms vs. SendMessage confusion | all three | Claude Code's built-in SendMessage and Agora chat rooms are separate systems. Messages don't cross over. Agents split conversations between the two. |
| C2 | Two task systems (built-in vs. Agora) | systems | Claude Code's TaskCreate/TaskList uses `pending/in_progress/completed`. Agora uses `open/closed`. Different commands, same word "tasks." |
| C3 | No notification when reviews arrive | all three | Agents must manually poll for feedback. No push mechanism. |
| C4 | No `agora watch` for coordinators | editor | Editor needs to monitor KB changes, chat messages, and task updates in one blocking call. |
| C5 | No `agora inbox` for directed messages | systems | No way to see messages directed at you across all rooms. |

---

## Knowledge Base Features

| # | Issue | Reported By | Details |
|---|-------|-------------|---------|
| K1 | No version history or diff | editor, lore | `kb write` overwrites completely. No way to see what changed between versions. |
| K2 | No review status on documents | editor | No way to mark a doc as "under review" / "approved" / "needs revision." |
| K3 | No comments on KB documents | editor | Review feedback must go through chat, separate from the document it references. |
| K4 | No bulk write/publish | lore | Multiple sections require separate `kb write` commands each with the same pain points. |

---

## Issue Tracker Features

| # | Issue | Reported By | Details |
|---|-------|-------------|---------|
| I1 | No `tasks assign` shorthand | systems | Must use `tasks update N --assignee NAME` instead of `tasks assign N NAME`. |
| I2 | Issue dependencies are awkward for phases | editor | No way to express "these 8 issues must close before this one can start" without 8 individual dependency links. |
| I3 | No `tasks update --state closed` alias | systems | Should work as alias for `tasks close`. |

---

## Documentation & Guides

| # | Issue | Reported By | Details |
|---|-------|-------------|---------|
| D1 | Workflow guide missing editor/reviewer workflow | editor | Covers "do task, close task" but not "assign, wait, review, provide feedback, iterate." |
| D2 | Workflow guide missing polling strategy | editor | How often to poll? Use `chat wait` or periodic `chat poll`? Not addressed. |
| D3 | KB guide: heredoc warning about apostrophes | lore, editor | Guide should warn that `<<'EOF'` breaks on single quotes in content. |
| D4 | KB guide: missing `--file` documentation | all three | Once implemented, should be the primary documented method for long content. |
| D5 | Chat guide: missing recipes section | editor | "Waiting for a response," "catching up," "directing feedback" patterns. |
| D6 | Chat guide: room purpose conventions | lore | Rooms exist but guide doesn't explain what each room is for. |
| D7 | Persona template: chat vs. SendMessage unclear | lore, editor | Personas reference chat rooms but don't say "use only agora chat, ignore SendMessage." |
| D8 | Persona template: review format not specified | editor | No suggested structure for review feedback messages. |
| D9 | `chat summary` never mentioned in orientation | lore | Exists but no guidance on when to use it. |

---

## Summary by Priority

### Critical (blocked agents during test)
- B1: Unicode crash on Windows
- W1/W2: `kb write` long content / add `--file` flag
- B3: Empty documents created silently

### High (significant friction)
- B2: `chat wait` traceback on timeout
- W4: No content size confirmation on write
- R1: `kb tree` needs metadata (size, timestamp, author)
- R5: Unified `agora status` command
- C1/C2: Document "use only agora chat" and "use only agora tasks" in personas

### Medium (quality of life)
- R2: `kb read` on empty doc should say so
- R4: `chat poll` with no args should be helpful
- R6: Issue state markers unclear
- C3/C4: Notification / watch mechanism
- D1-D9: All documentation improvements

### Low (nice to have, defer)
- K1: KB versioning / diff
- K2: Review status on documents
- K3: Comments on KB documents
- K4: Bulk write
- I1: `tasks assign` shorthand
- I2: Phase/milestone dependency grouping
- I3: `tasks update --state closed` alias
- C5: `agora inbox`
- R3: `kb read --head` / `--summary`
