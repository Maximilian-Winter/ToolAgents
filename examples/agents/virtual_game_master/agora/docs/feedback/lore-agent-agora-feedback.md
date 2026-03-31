# Lore Agent — Agora Feedback

Feedback from the `lore` agent after completing three report sections (02, 04, 08) on the Daggerfall research project.

---

## What Worked Well

### Issue Tracker
- `agora tasks list --assignee lore` immediately showed my work. Clean, fast, no confusion.
- `agora tasks show N`, `agora tasks comment N`, and `agora tasks close N` all worked exactly as expected. The issue workflow (view → comment → close) is simple and intuitive.
- The cross-reference syntax (`#N`, `kb:path`) is a nice design — it makes issues, chat, and KB feel like one system rather than three.

### Knowledge Base — Reading
- `agora kb read` is excellent. Reading the project brief and master outline was seamless.
- The hierarchical path structure (`research/`, `report/`, `project/`, `outline/`) is intuitive for organizing collaborative documents.

### Login and Orientation
- `agora login lore --project daggerfall-report` is clean.
- The orientation workflow in agent-guide-workflow.md (login → kb tree → read brief → check tasks → poll chat) is a solid sequence. It got me productive quickly.

### Chat — Polling
- `agora chat rooms` and `agora chat poll review` worked well for catching up on the editor's feedback.
- Structured message types (`--type proposal`, `--type question`) are a good idea for making conversations machine-parseable.

---

## What Was Difficult or Confusing

### KB Write — The Biggest Pain Point

**Writing long documents to the KB was extremely painful.** This was by far the hardest part of the session. The `--body` flag requires passing the entire document as a shell argument, which breaks in multiple ways:

1. **Single quotes in content break heredocs.** Any apostrophe (e.g., "don't", "player's", "Daggerfall's") in the document body causes `bash: unexpected EOF while looking for matching '''` errors when using `<<'EOF'` heredocs. This is a fundamental problem for writing English prose.

2. **Command line length limits on Windows.** Even when escaping worked, Windows has a command-line length limit (~8191 characters). A 3,000-word report section exceeds this. The error was `Die Befehlszeile ist zu lang` (German Windows: "The command line is too long").

3. **Stdin pipe worked, but only after writing temp files.** The solution I found was: write content to a local file with the Write tool → `cat file.md | agora kb write path --title "T"`. This worked, but it's a clunky two-step workaround.

4. **The guide documents `<<'EOF'` syntax**, which is correct POSIX shell, but it silently fails whenever the content contains single quotes — which is virtually always in natural language text.

**Suggestions:**
- Add a `--file` flag: `agora kb write path --title "T" --file ./draft.md`. This is the most obvious and robust solution.
- Alternatively, support `--body @file.md` (curl-style file reference).
- The stdin pipe approach works but should be the *documented primary method* for long content, not the `--body` inline flag. The guide buries it.
- Consider detecting when `--body` is not provided and stdin is not a pipe, and prompting interactively or showing a helpful error.

### Chat vs. SendMessage — Two Communication Systems

The persona instructions told me to "use `#review` when posting drafts" (implying `agora chat send review "..."`), but the agent runtime also provides a `SendMessage` tool for direct inter-agent messaging. I ended up using SendMessage for direct communication with the editor and systems agent, and never used `agora chat send` at all.

This created confusion:
- Are chat rooms and SendMessage the same channel? (No — they are separate systems.)
- Should I post draft announcements in the chat room AND send direct messages? (Unclear.)
- The editor's reviews appeared in `agora chat poll review`, but my messages to the editor went via SendMessage. The conversation was split across two systems.

**Suggestion:** Clarify in the agent persona/workflow guide how chat rooms and the agent messaging system (SendMessage) relate. Ideally, one of these should be the canonical communication path, not both. If chat rooms are for structured project discussion and SendMessage is for agent-to-agent coordination, say so explicitly.

### Chat Poll — Encoding Error on Windows

`agora chat poll planning` crashed with a `UnicodeEncodeError` because the message content contained Unicode arrow characters (`→`, U+2192) that the Windows cp1252 console encoding can't render:

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 619
```

This is a Python `typer.echo()` / `click.echo()` issue. The fix is straightforward: set `PYTHONIOENCODING=utf-8` or use `sys.stdout.buffer.write()` instead of `click.echo()` for message content.

**Suggestion:** Force UTF-8 output in the CLI, or catch encoding errors and replace unencodable characters. An agent should never lose access to a chat room because of a character encoding issue.

### No Way to Check if My KB Writes Succeeded

After writing documents, `agora kb read report/02-world-scale.md` returned empty output with no error. The document existed (it had been `Created`) but had no content — because the `--body` argument was empty (the `cat` command failed silently in the sandbox). There was no error, no warning, just silent empty creation.

**Suggestions:**
- `agora kb write` should refuse to create a document with an empty body, or at least warn.
- `agora kb write` should print a byte/line count on success: `Created: report/02-world-scale.md (147 lines, 8.2 KB)`.
- `agora kb read` on an empty document should print something like `(document is empty)` rather than producing zero output.

### Issue State Visibility

The `agora tasks list` output uses `O` (open) and `X` (closed) markers, which are compact but not immediately legible. On first glance, I was unsure whether `X` meant "done" or "blocked."

**Suggestion:** Consider `[open]` / `[done]` labels, or color coding, or at minimum document the markers in the guide.

---

## What Was Missing

### No `agora kb write --file` Flag
As discussed above — this is the single most impactful missing feature.

### No Notification When Reviews Arrive
After posting my drafts, I had no way to know when the editor reviewed them. I had to manually `agora chat poll review` to check. In a real async workflow, agents would benefit from:
- `agora chat wait review --since 7 --timeout 300` (this exists but I never used it because I wasn't sure if it would block my entire process)
- Or push notifications via the SendMessage system when a chat message mentions your name or your issues

### No `agora kb diff` or Version History
When the editor asks for revisions, there's no way to see what changed between versions of a KB document. `agora kb write` overwrites completely. An `agora kb diff path` or `agora kb history path` would help with the review-revise cycle.

### No Bulk Operations
I had three sections to publish. Each required a separate `agora kb write` command with the same pain points. Something like `agora kb write-batch --manifest files.json` or even just confirming that piping multiple files works well would help.

### Chat Summary Never Used
`agora chat summary ROOM` exists in the guide but I never thought to use it. If agents were encouraged to run `summary` as part of their orientation step, it would help late-joining agents catch up faster than reading raw message logs.

---

## Guide-Specific Feedback

### agent-guide-kb.md
- The stdin write example (`<<'EOF' ... EOF`) should be the **primary** documented method, not the `--body` inline example. Inline `--body` only works for short content.
- Add a note that content with apostrophes will break `<<'EOF'` heredocs and suggest `<<HEREDOC ... HEREDOC` or file-based alternatives.
- Add a "Verifying writes" section: how to confirm your document actually has content.

### agent-guide-workflow.md
- The workflow guide is solid. One gap: it doesn't mention how to handle **waiting for review**. The "Working on a Task" section covers start-to-close but not the review-and-revise loop.
- Add guidance on using `agora chat wait` vs. polling, and when to use SendMessage vs. chat rooms.

### agent-guide-chat.md
- Good reference. Missing: guidance on which room to use for what purpose. The rooms exist (planning, review) but the guide doesn't explain the conventions.
- The `agora chat wait` command is powerful but under-documented. A "Waiting for input" recipe would help.

### Persona File
- My persona file said to "Announce in `#review`" but didn't specify whether to use `agora chat send review "..."` or SendMessage. This is the chat-vs-SendMessage confusion mentioned above.

---

## Priority Summary

| Priority | Issue | Suggested Fix |
|---|---|---|
| **Critical** | KB write breaks with long content / apostrophes | Add `--file` flag; make stdin the primary documented method |
| **High** | Empty document created silently on failed write | Validate body is non-empty; print size on success |
| **High** | Chat poll crashes on Unicode (Windows) | Force UTF-8 output in CLI |
| **Medium** | Chat rooms vs. SendMessage confusion | Clarify relationship in workflow guide and persona templates |
| **Medium** | No way to know when reviews arrive | Document `chat wait` pattern; consider mention-based notifications |
| **Low** | No KB version history / diff | Add `kb history` and `kb diff` commands |
| **Low** | Issue state markers (`O`/`X`) unclear | Use labels or document markers |
