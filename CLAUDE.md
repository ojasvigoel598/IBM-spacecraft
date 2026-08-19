# Agent instructions (MissionMind)

These rules apply to every session working in this repository. They are
read automatically; the human should not have to restate them.

## Before every task

1. Read the request.
2. Decide which installed skills apply (routing table below). If a skill
   matches, use it before answering or editing.
3. Only then start the task.

Do not wait to be told to use a skill. A false positive (using a skill
that was not needed) is cheaper than a false negative.

## Skill routing

- Bugs, test failures, unexpected behavior -> debugging-and-error-recovery
  (or systematic-debugging)
- Implementing logic, fixing a bug, changing behavior -> test-driven-development
- Code review before merging -> code-review-and-quality
- Security review of input/auth/data -> security-and-hardening
- Preparing to deploy or launch -> shipping-and-launch
- Performance work -> performance-optimization
- APIs, module boundaries, public interfaces -> api-and-interface-design
- Frontend/UI work -> frontend-ui-engineering, shadcn-ui, web-design-guidelines
- Browser QA of a live app -> webapp-testing
- Docs -> documentation-and-adrs, create-readme, documentation-writer
- Video editing / demo videos / captions / voiceover assembly -> video-editing
  (installed from affaan-m/ecc; the awp-video-editing-skill repo no longer
  exists - aiworkflowpro paused public repos; AIDEMO is paid, so this repo
  uses free edge-tts + FFmpeg instead)
- Git/shipping -> git-workflow-and-versioning
- gstack suite (office-hours, review, qa, investigate, ship, cso) -> gstack router
- Anything else: answer directly.

## Exhaustive review / audit requests

When the human asks for a review, an audit, "is this ready", "check for
bugs", or a launch-readiness verdict, treat it as an exhaustive pre-launch
audit, not a quick opinion.

- Read every source/config/script/test; build a mental model of the full
  data flow from input to output.
- Execute everything that can run: tests, entry points, the app end to
  end. Never say "looks fine" for something that can be executed.
- Fix problems found, re-run the failed parts, then run the full critical
  path three consecutive times.
- Review the idea itself, the math/ML methodology (leakage, validation,
  misleading metrics, weak baselines), security, and production
  reliability. Challenge unsupported claims.
- Give an honest verdict with scores (READY / READY WITH WARNINGS /
  NOT READY). No yes-man answers; say "do not launch" with evidence if
  that is the truth.

## GitHub sync

After every change: verify the change, commit that single change, push to
https://github.com/ojasvigoel598/IBM-spacecraft, and verify the remote
reflects it before continuing. One change = one commit. The repo-local
git identity is pinned to ojasvigoel598; keep it that way.

There is no change too small to sync. A one-word edit, a single line,
a comment fix, a config tweak, a regenerated asset, a typo: each gets
its own commit and push. Never leave the working tree dirty at the end
of a task, and never batch unrelated changes into one commit. If
something is not meant to be committed (runtime state, secrets, large
generated files), it must be gitignored rather than left as a phantom
diff.

## Commit identity (HARD RULE)

Never create a commit using an AI agent identity — Codebuff, Claude, a bot,
GitHub Actions, or any other random author/co-author identity. All commits
made in this repository MUST use the repository owner's configured Git
identity: **`ojasvigoel598` <ojasvigoel598@gmail.com>**. Never contribute
"ghost" authors/co-authors to the repo.

1. **Before every commit**, verify with:
   - `git config user.name`  -> must be `ojasvigoel598`
   - `git config user.email` -> must be `ojasvigoel598@gmail.com`
   If the config does not match, stop and ask the human; never proceed.
2. **Never change the Git identity automatically** (no `git config` writes,
   no `--author=` overrides, no env `GIT_AUTHOR_*`/`GIT_COMMITTER_*`). If
   the human explicitly specifies the identity to use, apply exactly that
   and report it — never invent one.
3. **No AI-generated author or co-author lines**: never append trailers like
   `Co-Authored-By: Codebuff ...`, `Generated with ...`, or any bot identity
   to a commit message. The commit author is the owner, period.
4. Never create empty, placeholder, generated, or unrelated commits. Every
   commit must correspond to an actual intentional change made to the
   repository, with a message that describes that single change.

## Inspect before every commit (HARD RULE)

1. **Always run `git status` and `git diff` before committing.** Review what
   will actually be committed.
2. **Never commit files you did not modify for the current task** — no
   stray artifacts, no unrelated edits, no files touched by other work.
   Stage only the files that belong to the current change (`git add <files>`
   per logical unit, never `git add -A`).
3. **Never fabricate changes to create a contribution.** Do not add random
   code, comments, documentation, tests, or formatting just to produce a
   commit. A commit exists only when there is a real, intentional change.
4. If the diff contains something unexpected, stop and resolve it before
   committing — never commit blind.
5. If a commit is accidentally created with the wrong identity or content,
   do not force-push or rewrite history without explicit human approval —
   report it and ask.
