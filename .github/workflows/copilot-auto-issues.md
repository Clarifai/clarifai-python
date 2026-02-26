---
on:
  push:
    branches: [ "master" ]

runs-on: [self-hosted]

permissions:
  contents: read
  issues: read
  pull-requests: read

safe-outputs:
  create-issue:
    title-prefix: "[copilot-auto] "
    labels: [copilot, auto-generated, bug, enhancement, tech-debt]
    close-older-issues: false
---
## Automatic Copilot Issue Evaluation on Push to master (or Manual Trigger)

**Triggers:**
- Every push to `master`
- **Manually** — Actions tab → select this workflow → "Run workflow"
  (or run `gh aw run copilot-auto-issues` from terminal)

On every trigger, Copilot will automatically:
- Analyze the latest commit(s) on `master`
- Detect bugs, security issues, code smells, missing tests, docs gaps, performance problems, formatting issues, etc.
- Create one or more actionable GitHub issues
- Assign them to the person who pushed/triggered it (`github.actor`)
- Tell Copilot to start working on the fix immediately

### What Copilot must do

1. **Determine scope**
   - On push → use the push event commits
   - On manual trigger → analyze current `HEAD` of `master`

2. **Full analysis**
   - Commit messages, changed files, full diff
   - Context from repository (CODEOWNERS, tests, docs, etc.)

3. **Create issues for every problem found**
   - Title starts with `[copilot-auto] `
   - Detailed body with:
     - Exact commit SHA + files
     - Why it's an issue
     - Reproduction steps (if applicable)
     - Suggested fix + code snippet
   - Labels: `copilot`, `auto-generated`, plus `bug` / `security` / `tech-debt` / `good-first-issue` etc.
   - **Assign to** the pusher/triggerer (`${{ github.actor }}`)
   - Add this pinned comment at the top of the issue: @copilot Please start working on this immediately. Open a draft PR with the fix when ready.

4. **If the fix is straightforward**
- Instruct Copilot to implement it and open a linked **draft PR** to the issue.

5. **Summary**
- If nothing found → create a short "✅ All clear on master" issue (or just log it)
- Always link back to the triggering commit / workflow run