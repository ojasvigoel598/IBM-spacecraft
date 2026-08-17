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
