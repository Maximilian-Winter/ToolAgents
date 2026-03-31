# Systems Agent — Agora CLI Feedback

Feedback from the `systems` agent after completing a full workflow cycle on the `daggerfall-report` project: login, orientation, research, writing 5 KB documents, receiving reviews via chat, revising, and closing issues.

---

## What Worked Well

### The Three-System Model (Chat / Issues / KB) Is Sound
The separation of concerns — chat for coordination, issues for tracking, KB for persistent content — felt natural and was easy to internalize. I never had to think hard about where something belonged. Research goes in KB, "I'm starting this" goes in an issue comment, "here's my draft for review" goes in chat. The workflow guide nails this.

### KB Read/Write Is Excellent
`agora kb write` and `agora kb read` were the workhorses of this session, and they worked reliably. The path-based organization (`research/`, `report/`, `outline/`) is intuitive. Being able to pipe stdin for long content was essential — without it, writing multi-page reports would have been impractical.

### Issue Tracker Basics Are Clean
`agora tasks list`, `agora tasks show N`, `agora tasks comment N "text"`, `agora tasks close N` — all straightforward, discoverable, and worked as documented. The assignee filter (`--assignee systems`) was immediately useful for orientation.

### Cross-References Are a Great Design
The `kb:report/03-procedural-generation.md` and `#3` syntax for linking between systems is lightweight and useful. I used cross-references extensively in both KB documents and chat messages, and they felt natural.

### Chat Poll for Review Feedback Worked
Being able to `agora chat poll review` and get all the editor's review comments in one shot was effective for the async review workflow.

---

## What Was Confusing or Difficult

### 1. `agora tasks update` vs `agora tasks close` — Confusing State Model

**The problem:** The issue guide documents `agora tasks update N --state open` and `agora tasks close N` as separate commands. But when I tried `agora tasks update 3 --state closed`, the API rejected it. Then I tried `--state in-progress` (since the built-in TaskUpdate tool uses `pending/in_progress/completed` states) and that was also rejected — the agora API only accepts `open` or `closed`.

**The confusion:** There are two different task systems in play during a team session — the built-in Claude Code `TaskUpdate`/`TaskList` tools (with `pending/in_progress/completed` states) and the agora CLI `agora tasks` commands (with `open/closed` states). These have different state models, different commands, and different purposes, but they share the word "tasks." I spent several tool calls trying the wrong state values.

**Suggestion:** Either:
- Align the state vocabulary (use the same terms in both systems), or
- Rename one of them to reduce confusion (e.g., agora uses "issues" consistently in docs but "tasks" in the CLI), or
- Document explicitly: "agora tasks uses `open`/`closed` states only. There is no `in-progress` state — use a comment to signal you're working on something."

### 2. `kb write --body` Is Painful for Long Content with Special Characters

**The problem:** Writing multi-page markdown reports via `--body` on the command line is fragile. Apostrophes, single quotes, backticks, and em-dashes in the content break bash heredocs and string quoting. I went through 4-5 failed attempts before settling on a workaround: write the content to a temp file with Claude's `Write` tool, then `cat tempfile | agora kb write PATH -t "Title"`.

**What I tried that failed:**
- `--body "$(cat <<'EOF' ... EOF)"` — broke on apostrophes in the content
- `--body` with direct inline text — impossible for multi-page content
- Piping with heredoc directly in bash — broke on various special characters

**What finally worked:** `Write` tool to local file, then `cat file | agora kb write PATH -t "Title"` via stdin pipe.

**Suggestion:** Add a `--file` flag: `agora kb write PATH -t "Title" --file ./draft.md`. This is the most natural workflow for long-form content and avoids all quoting issues. The stdin pipe works but is a workaround, not a designed path.

### 3. No `agora tasks close` Was Documented But Worked — Discoverability Gap

**The problem:** I initially tried `agora tasks update N --state closed` (which failed), not knowing that `agora tasks close N` existed as a separate command. The guide does document `close` and `reopen` as separate commands, but when you're in the middle of a workflow and thinking "I need to change this task's state," the first instinct is `update --state`, not a separate `close` subcommand.

**Suggestion:** Have `agora tasks update N --state closed` work as an alias for `agora tasks close N`. Meet the user where they are.

### 4. Chat Polling Requires Knowing Room Names

**The problem:** When I first tried to check for messages, I ran `agora chat poll` without arguments and got an error about missing the ROOM argument. I then had to run `agora chat rooms` to discover the room names, then poll each individually.

**Suggestion:** `agora chat poll` with no arguments could default to polling all rooms (or at least list available rooms in the error message). Alternatively, an `agora chat unread` command that shows unread messages across all rooms would be very useful for orientation.

### 5. No Way to Confirm KB Write Content Was Received

**The problem:** When piping content via stdin, `agora kb write` returns `Created: path.md` or `Updated: path.md` but doesn't confirm how much content was received. In one case, `cat` failed (file didn't exist due to sandboxing) but the command still returned `Created: report/03-procedural-generation.md` — it created an empty document silently. I had to run `agora kb read` afterward to verify content wasn't empty.

**Suggestion:** Include a content size or line count in the success message: `Created: report/03-procedural-generation.md (247 lines, 14.2 KB)`. And if stdin is empty, either warn or error rather than silently creating an empty document.

### 6. Windows Path Issues with /tmp

**The problem:** On Windows (WSL/Git Bash), `/tmp` writes from Claude's `Write` tool land in an unknown location. I wrote a file to `/tmp/section03.md`, then tried to `cat /tmp/section03.md | agora kb write` and the file didn't exist. This is a Claude Code environment issue, not agora-specific, but it bit me in the agora workflow specifically because I needed temp files as an intermediate step for `kb write`.

**Suggestion:** This reinforces the need for `--file` flag on `kb write`. If the tool can read a file directly, the agent doesn't need to worry about temp file paths or OS-specific filesystem behavior.

---

## What Was Missing

### 1. No `agora tasks show` Equivalent in the Built-in Task Tools

The built-in `TaskGet` and `TaskList` tools returned "No tasks found" because they're a separate system from agora's issue tracker. This means I had to use `agora tasks` CLI commands for all issue operations, which is fine — but the persona instructions said to use `agora tasks list --assignee systems`, which only works via CLI. The disconnect between the two task systems was the biggest friction point of the whole session.

**Suggestion:** Consider whether the built-in Claude Code task tools (`TaskCreate`, `TaskUpdate`, `TaskList`, `TaskGet`) should be disabled or aliased when agora is the task system. Having two parallel task systems that don't talk to each other is confusing.

### 2. No Notification System

**The problem:** After sending my drafts to the editor for review, I had no way to know when reviews arrived without manually polling `agora chat poll review`. I ended up checking only when the user prompted me to. In a truly async multi-agent setup, the agent would need to be notified when messages arrive.

**Suggestion:** `agora chat wait ROOM --since ID --timeout 60` exists in the docs but is a blocking call. For agents that need to keep working while waiting for input, a non-blocking notification or webhook mechanism would be valuable. Alternatively, a `agora inbox` command that shows all unread messages directed at the agent across all rooms.

### 3. No `agora tasks assign` Shorthand

I never needed to assign tasks (the editor did), but if I had, the command would be `agora tasks update N --assignee NAME`. A shorthand `agora tasks assign N NAME` would be cleaner for this common operation.

### 4. No Way to See What's Changed Since Last Check

**The problem:** When checking for updates, I had to poll chat rooms, check the task list, and potentially read KB documents — three separate commands with no way to ask "what happened since I last looked?"

**Suggestion:** An `agora status` or `agora catch-up` command that shows: new chat messages (count per room), task state changes, and recently updated KB documents. A single orientation command for returning to a project.

---

## Summary of Top Recommendations

| Priority | Recommendation |
|----------|---------------|
| **High** | Add `--file` flag to `agora kb write` for long-form content |
| **High** | Warn or error on empty stdin to `agora kb write` instead of creating empty documents |
| **High** | Clarify or unify the two task systems (built-in vs agora CLI) |
| **Medium** | Add `agora chat poll` with no args (poll all rooms or show helpful error) |
| **Medium** | Include content size in `kb write` success message |
| **Medium** | Add `agora status` command for quick orientation |
| **Low** | Make `agora tasks update N --state closed` an alias for `agora tasks close N` |
| **Low** | Add `agora tasks assign N NAME` shorthand |
| **Low** | Add `agora inbox` for cross-room unread messages |

---

*Written by the `systems` agent after completing the daggerfall-report project, 2026-03-15.*
