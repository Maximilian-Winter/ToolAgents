# Editor Agent — Agora Feedback

Feedback from the `editor` agent after coordinating the full Daggerfall report project: defining the outline, creating all issues, reviewing 8 drafts from `systems` and `lore`, writing 2 sections, and managing the project to completion.

---

## What Worked Well

### The Three-System Architecture (KB + Issues + Chat)

The separation of concerns between knowledge base, issue tracker, and chat rooms is sound. Each system has a clear purpose and they complement each other well:
- **KB** for persistent artifacts (outline, research, report sections)
- **Issues** for tracking work assignments and completion
- **Chat rooms** for coordination (#planning) and feedback (#review)

The cross-reference syntax (`kb:report/03-procedural-generation.md`, `#5`) ties them together effectively. I used these references constantly in reviews and they made the feedback feel connected to the work.

### Issue Tracker — Clean and Functional

Creating issues with `agora tasks create`, assigning with `--assignee`, and closing with `agora tasks close` all worked as expected. The workflow for the editor role — create issues, assign them, monitor completion — was straightforward.

The issue list output is compact and scannable. I could quickly see which sections were open vs. closed and who owned what.

### KB Tree — Essential for Monitoring

`agora kb tree` was my most-used command. As the coordinating agent, I needed to constantly check what new documents had appeared — new research notes, new draft sections. The tree view gave me a snapshot of project progress at a glance. This is a well-designed command.

### Chat Rooms for Reviews

Posting structured reviews in `#review` with `--to systems` or `--to lore` worked well. The message IDs and `--since` flag let me avoid re-reading old messages. The `--type` flag (question, consensus, proposal) is a good idea for giving messages semantic weight.

### Login and Orientation Flow

The orientation sequence in agent-guide-workflow.md (login, kb tree, read brief, check tasks, poll chat) is solid. It got me productive within minutes.

---

## What Was Difficult or Confusing

### Unicode / Encoding Crashes on Windows (Critical Blocker)

This was the single biggest obstacle in the entire session. The `agora chat poll` and `agora kb read` commands crash hard on Windows when content contains non-ASCII characters:

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 619
```

This happened because:
1. My own message in #planning contained `→` arrow characters (from the master outline)
2. The `lore` agent's KB documents contained Unicode subscript characters (`₀` in the leveling formula)
3. The `lore` agent's report drafts contained em dashes encoded as multi-byte UTF-8

The crash in `typer.echo()` / `click.echo()` completely blocked me from using `agora chat poll planning` and `agora kb read` for several documents. I had to work around it by:
- Using `--since N` to skip past the offending message in chat
- Using `curl` directly against the HTTP API + Python JSON parsing to read KB documents

**This workaround consumed significant time and attention.** An agent should never have to bypass the CLI to read its own project's documents.

**Fix:** Force `PYTHONIOENCODING=utf-8` at CLI startup, or use `sys.stdout.buffer.write()` instead of `click.echo()`. This is a one-line fix in the CLI that would eliminate the entire class of errors.

### KB Write — Long Content Is Painful

Like the `lore` agent, I struggled with writing long documents. The `--body` flag works for short content but not for multi-thousand-word report sections. Heredocs with `<<'EOF'` break on apostrophes in natural English text. My workarounds:
- Write to a temp file first, then `--body "$(cat temp.md)"` — worked for the Introduction
- The Legacy section's content broke even the `$(cat)` approach due to shell quoting, so I had to write a temp file and use `--body "$(cat file)"` after cleaning quotes

**The `--file` flag suggested by lore is the correct solution.** This should be priority one.

### No Diff / Change Detection on KB Documents

As editor, my core job was detecting when new content appeared and reviewing it. But I had no efficient way to do this:

- `agora kb tree` shows document existence but not whether content changed
- `agora kb read` returns the full document every time — no way to see what's new
- Documents that were created as empty placeholders (e.g., `report/03-procedural-generation.md` appeared in the tree with 0 bytes) were indistinguishable from completed drafts until I read them

I ended up using `curl` to check content length (`len(content)`) to determine whether a document was a placeholder or a real draft. This should not require HTTP API access.

**Suggestions:**
- `agora kb list` or `agora kb tree` should show document size or last-modified timestamp
- `agora kb read --head 5` or `agora kb read --summary` to quickly preview without loading the full document
- `agora kb diff path` to see what changed since last read

### Chat Wait — Timeout Errors Are Noisy

`agora chat wait planning --since 2 --timeout 60` works correctly (blocks, then returns) but on timeout it throws a full Python traceback (`ReadTimeout: timed out`) with ~50 lines of stack trace. This is alarming when it's actually expected behavior — no new messages simply means nothing happened.

**Fix:** Catch the timeout exception in the CLI and print a clean message: `No new messages (timed out after 60s)`. The full traceback suggests a bug, not a normal condition.

### No Notification System for the Editor Role

The editor workflow is fundamentally reactive: wait for drafts, review them, provide feedback, wait for revisions. But agora provides no push mechanism — I had to manually poll:
- `agora chat poll review` — any new messages?
- `agora kb tree` — any new documents?
- `agora tasks list` — any issues closed?

In a real multi-agent system, the editor should be notified when:
- A document in `report/` is created or updated
- An issue assigned to another agent is closed
- A message in #review is directed to `editor`

**Suggestion:** Consider a `agora watch` command that monitors multiple channels: `agora watch --kb "report/*" --chat review --tasks --timeout 300`. This would let the editor block until something relevant happens, rather than polling three separate systems.

### Chat Rooms vs. SendMessage — Architectural Confusion

The agent runtime provides `SendMessage` for direct inter-agent messaging, while agora provides `agora chat send` for room-based discussion. These are separate systems that don't interact:
- Messages sent via `SendMessage` don't appear in `agora chat poll`
- Messages sent via `agora chat send` don't trigger agent notifications

My persona said to use `#planning` and `#review` (chat rooms), so I used `agora chat send` for all communication. But this means my reviews were visible to agents only if they polled the room — there was no notification that a review was waiting.

**This is the most important architectural question for agora:** Are chat rooms the primary communication channel, or is SendMessage? Currently the answer is "both, sort of, depending on context," which creates confusion for both the agent writing the persona and the agent following it.

**Suggestion:** Either:
1. Bridge them: `agora chat send review "..." --to lore` should also trigger a SendMessage notification to `lore`
2. Or clearly separate them: chat rooms for persistent team discussion (async), SendMessage for direct pings (sync), with documentation explaining when to use which

### Empty Document Creation Is Silent

`agora kb write report/03-procedural-generation.md --title "..." --body ""` silently creates an empty document with no warning. I later found this document in `agora kb tree` and assumed it was a completed draft, only to discover it had zero content when I read it via the API.

**Fix:** Refuse to create a document with an empty body, or at minimum print a warning: `Warning: document created with empty body`.

---

## What Was Missing

### KB Document Metadata in Tree/List

`agora kb tree` shows path and title only. For the editor role, I need:
- **Size** (character count or line count) — to distinguish placeholders from real drafts
- **Last modified** — to know when content was updated
- **Author** — to know who wrote/updated it

Something like:
```
report/
  03-procedural-generation.md — "Procedural Generation" (13.6k, systems, 2h ago)
  04-faction-systems.md       — "Faction Systems"       (0 bytes, editor, 5h ago)
```

### KB Read with Section Targeting

`agora kb read "path#Section Name"` is documented but I never used it because I needed to read entire drafts for review. What I actually wanted was the opposite: `agora kb read path --summary` or `--head N` to quickly preview a document without loading all 15,000 characters.

### Issue Dependencies

The issue tracker supports `agora tasks add-dependency`, but there's no way to express "issue #1 and #10 are blocked until #2-#9 are all closed." I wanted to mark my Introduction and Legacy sections as blocked by all 8 other sections, but adding 8 individual dependency links felt excessive. A milestone or phase concept would help.

### Project-Level Dashboard

As editor, I wanted a single command that showed me everything: open issues, recent KB changes, unread chat messages. Something like:
```
agora status
```
That combines `tasks list --state open`, `kb tree` (with recent changes highlighted), and unread message counts per room.

### Review Workflow Support

The review-revise cycle is central to the editor workflow but has no dedicated tooling:
- No way to mark a document as "under review" vs. "approved" vs. "needs revision"
- No way to attach review comments directly to a KB document (I had to use chat rooms, which separates the feedback from the artifact)
- No way for the author to see "what feedback exists for my document"

**Suggestion:** Consider `agora kb review path --status approved` or `agora kb comment path "feedback text"` to attach review state and feedback directly to documents.

---

## Guide-Specific Feedback

### agent-guide-workflow.md
- **Strong overall.** The orientation flow is good, the "Working on a Task" section is clear.
- **Missing:** The editor/reviewer workflow. The guide covers "pick up a task, do it, close it" but not "assign tasks, wait for work, review artifacts, provide feedback, iterate." Add a "Coordinating Work" or "Reviewing Work" section.
- **Missing:** Guidance on polling strategy. How often should an agent poll? Should they use `chat wait` or periodic `chat poll`? The guide doesn't say.

### agent-guide-kb.md
- **Missing:** Any mention of document size limits, encoding considerations, or the `--file` flag (which doesn't exist yet but should).
- **Improve:** The stdin heredoc example should warn about apostrophes breaking `<<'EOF'`. Suggest `<<HEREDOC` (unquoted, with variable expansion disabled via escaping) or file-based input as primary methods.
- **Add:** A "Reading large documents" section noting that very long documents will produce a lot of output and suggesting `#Section` targeting for focused reads.

### agent-guide-chat.md
- **Good reference** but could use a "Recipes" section:
  - "Waiting for a response": use `chat wait --since ID --timeout N`
  - "Catching up after being away": use `chat poll ROOM --since LAST_READ_ID`
  - "Directing feedback to a specific agent": use `--to NAME`
- **Missing:** Note that `chat wait` timeout produces a noisy traceback on Windows (should be fixed in CLI, but document as known issue until then).

### Persona Template Feedback
- The editor persona references `#planning` and `#review` without explaining whether these are `agora chat` rooms or SendMessage channels. This should be explicit.
- The persona says "Post feedback in `#review`" but doesn't specify the format. A suggested review template (what to cover, how to structure feedback, how to indicate approval vs. revision needed) would make reviews more consistent across projects.

---

## Priority Summary

| Priority | Issue | Suggested Fix |
|---|---|---|
| **Critical** | Unicode crashes on Windows (chat poll, kb read) | Force UTF-8 output in CLI |
| **Critical** | KB write breaks with long content / apostrophes | Add `--file` flag |
| **High** | Chat wait timeout throws full traceback | Catch timeout, print clean message |
| **High** | Empty documents created silently | Validate non-empty body; print size on success |
| **High** | No change detection for KB documents | Add size/timestamp to `kb tree`; add `kb diff` |
| **Medium** | Chat rooms vs. SendMessage confusion | Clarify or bridge the two systems |
| **Medium** | No notification/watch mechanism for coordinators | Add `agora watch` or event subscription |
| **Medium** | No review workflow tooling | Add `kb review`/`kb comment` commands |
| **Low** | No project dashboard command | Add `agora status` combining tasks + kb + chat |
| **Low** | Issue dependency ergonomics | Support milestone/phase grouping |
| **Low** | Issue state markers (`O`/`X`) unclear at first glance | Use text labels or document the markers |
