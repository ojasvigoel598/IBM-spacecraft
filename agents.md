# Agent Rules

Rules for any agent (Codebuff, Claude Code, Copilot, etc.) working in this repository.

## Commits — every logical change, immediately

1. **Commit after every single logical or incremental change, no matter how small.**
   Never wait to bundle updates into a major feature.
2. A "logical change" is one unit of work: a one-line fix, an import added, a test
   added, a config tweak, a doc sentence, a regenerated artifact. Commit it.
3. Never batch unrelated changes into one commit. If several related edits landed
   together on disk, split them into logical commits (`git add <files>` per unit)
   where practical.
4. Commit message must describe the single change (`import xx`, `fix: ...`,
   `test: ...`), matching the repository's existing message style.
5. Keep the branch clean and synced: `git push` after each commit when the remote
   is connected and pushing is expected.

## Commit identity (HARD RULE)

1. Never create a commit using an AI agent identity — Codebuff, Claude, a bot,
   GitHub Actions, or any other random author/co-author identity. Never
   contribute "ghost" authors/co-authors to the repo. All commits MUST be
   authored by the repository owner's identity:
   **`ojasvigoel598` <ojasvigoel598@gmail.com>**.
2. Before every commit, run and check:
   - `git config user.name`  -> must be `ojasvigoel598`
   - `git config user.email` -> must be `ojasvigoel598@gmail.com`
   If either does not match, STOP and ask the human. Never proceed.
3. Never change the Git identity automatically (no `git config` writes, no
   `--author=` overrides, no `GIT_AUTHOR_*` / `GIT_COMMITTER_*` env vars). If
   the human explicitly specifies the identity, apply exactly that and report
   it — never invent one.
4. Never add AI author/co-author trailers to commit messages (no
   `Co-Authored-By: Codebuff ...`, no `Generated with ...` lines).
5. Never create empty, placeholder, generated, or unrelated commits. Every
   commit corresponds to an actual intentional change, one change per commit.

## Inspect before every commit (HARD RULE)

1. Always run `git status` and `git diff` before committing; review what will
   be committed.
2. Never commit files you did not modify for the current task. Stage only the
   files that belong to the change (`git add <files>` per unit, never
   `git add -A`).
3. Never fabricate changes to create a contribution: no random code,
   comments, documentation, tests, or formatting added just to produce a
   commit. A commit exists only when there is a real, intentional change.
4. If the diff contains something unexpected, stop and resolve it first.
5. If the wrong identity or content was ever used, do not force-push or
   rewrite history without explicit human approval — report it and ask.
