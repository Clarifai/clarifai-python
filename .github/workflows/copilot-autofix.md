---
on:
  workflow_run:
    workflows: ["Run tests"]
    types: [completed]
    branches: [master]
  workflow_dispatch:

permissions:
  contents: read
  pull-requests: read
  actions: read

safe-outputs:
  add-comment:
    max: 10
---
## Auto-Fix Failing Tests on Existing PRs

**Trigger:** Fires when the "Run tests" workflow completes (on any branch).

If the conclusion is `failure` AND the run is associated with a pull request,
Copilot should analyze the failures and post fix suggestions **directly on
the existing PR** — no new PR, no new issue.

If the conclusion is `success`, do nothing and exit.

### Instructions for Copilot

1. **Gate on failure**
   - Check `github.event.workflow_run.conclusion`
   - If NOT `failure` → exit immediately, log "Tests passed — nothing to fix"

2. **Find the associated PR**
   - Check `github.event.workflow_run.pull_requests` for the PR(s) linked to this run
   - If no PR is associated (e.g. a direct push to main), exit — this workflow
     only handles PRs
   - Note the PR number, branch name, and head SHA

3. **Get failure details**
   - Download ALL `test-results-*` artifacts from the triggering run
   - Parse each `test-results.xml` (JUnit XML) to find:
     - Which tests failed (full `module::class::test` path)
     - Error messages and stack traces
     - Which OS + Python version failed
   - Read `test-output.txt` for additional context
   - Get the commit SHA from `github.event.workflow_run.head_sha`

4. **Analyze the failures**
   - Checkout the repo at the failing commit on the PR branch
   - For each failing test, trace the error back to the source code
   - Determine the root cause — is it a bug in the source, a test that needs
     updating, a missing dependency, or a platform-specific issue?
   - Identify the exact file(s) and line(s) that need to change

5. **Post fix suggestions on the existing PR**
   - Add a comment on the PR with a summary:
     - Link to the failed workflow run: `${{ github.event.workflow_run.html_url }}`
     - How many tests failed and on which platforms
     - Brief root cause for each failure group
   - For each fix, use GitHub suggestion blocks so the author can accept with
     one click. Format each suggestion like:

     ````
     ```suggestion
     corrected code here
     ```
     ````

   - If multiple tests fail for the same root cause, group them into one suggestion
   - If a fix spans multiple files or is too complex for a suggestion block,
     describe the fix clearly in the comment with code snippets and file paths
   - Always explain WHY the change fixes the test

6. **Scope rules**
   - Fix ONLY what's needed to make the failing tests pass
   - Do not suggest refactoring unrelated code
   - Do not weaken test assertions to make them pass — fix the source
   - If the fix is unclear or risky, say so in the comment
   - If there are merge conflicts (like `clarifai/__init__.py`), note them but
     do not attempt to resolve — just mention the conflicts need manual resolution
