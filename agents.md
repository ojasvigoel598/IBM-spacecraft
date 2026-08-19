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
   GitHub Actions, or any other random author/co-author identity. All commits
   MUST be authored by the repository owner's configured Git identity:
   `ojasvigoel598`.
2. Before every commit, run and check:
   - `git config user.name`  -> must be `ojasvigoel598`
   - `git config user.email` -> must be the owner's configured email
   If either does not match, STOP and ask the human. Never proceed.
3. Never change the Git identity automatically (no `git config` writes, no
   `--author=` overrides, no `GIT_AUTHOR_*` / `GIT_COMMITTER_*` env vars).
4. Never add AI author/co-author trailers to commit messages (no
   `Co-Authored-By: Codebuff ...`, no `Generated with ...` lines).
5. Never create empty, placeholder, generated, or unrelated commits. Every
   commit corresponds to an actual intentional change, one change per commit.
6. If the wrong identity was ever used, do not rewrite or force-push history
   without explicit human approval — report it and ask.
