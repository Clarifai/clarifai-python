---
on:
  workflow_run:
    workflows: ["Run tests", "CodeQL", "dynamic"]
    types: [completed]
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  issues: read
  pull-requests: read
  actions: read

safe-outputs:
  create-issue:
    title-prefix: "[copilot-auto] "
    labels: [copilot, auto-generated, bug]
    close-older-issues: false
    max: 10
---
## Automatic Copilot Issue Creation on CI Failures

**Triggers:**
- **Automatically** when any of these workflows **completes**:
  - **Run tests** — pytest matrix across 3 OSes × 2 Python versions
  - **CodeQL** — security / code scanning analysis
  - **dynamic** — PyPI package publish
- **Manually** — Actions tab → select this workflow → "Run workflow"
  (or run `gh aw run copilot-auto-issues` from terminal)

**Important:** This workflow fires whenever one of the watched workflows finishes.
It checks `github.event.workflow_run.conclusion` and only creates issues when
the conclusion is `failure`. Use `github.event.workflow_run.name` to determine
which workflow failed and tailor the response accordingly.

On trigger, Copilot will automatically:
- Determine which workflow failed (tests, CodeQL, or PyPI publish)
- Gather failure details from artifacts or workflow logs
- Create actionable GitHub issues for each distinct failure
- Assign them to the person whose push caused the failure (`github.actor`)
- Tell Copilot to start working on the fix immediately

### What Copilot must do

1. **Gate on failure** (skip if CI passed)
   - Check `github.event.workflow_run.conclusion`
   - If conclusion is NOT `failure`, skip all steps and log "CI passed — nothing to do"
   - On manual trigger → fetch the latest runs for all three watched workflows via the API

2. **Identify which workflow failed**
   - Read `github.event.workflow_run.name` to determine the source:
     - `"Run tests"` → test failure path (step 3a)
     - `"CodeQL"` → security / code scanning failure path (step 3b)
     - `"dynamic"` → PyPI publish failure path (step 3c)
   - Identify the commit SHA and branch from `github.event.workflow_run.head_sha`

3a. **Test failures ("Run tests")**
   - Download ALL `test-results-*` artifacts from the triggering workflow run
     (one per matrix combination, e.g. `test-results-ubuntu-latest-py3.11`)
   - Parse each `test-results.xml` (JUnit XML) to extract:
     - Failed test names and their module/file paths
     - Error messages and stack traces
     - Which OS + Python version the failure occurred on
   - Also read each `test-output.txt` for full pytest verbose output
   - Note which platforms passed vs failed — a test that only fails on Windows
     or macOS is a platform-specific bug and should be labeled accordingly
   - Create one issue per distinct failure (or group related failures):
     - Labels: `test-failure`, plus `platform-specific` if OS-specific
     - Body includes: exact test name(s), error + stack trace, OS/Python matrix, root cause, fix snippet

3b. **CodeQL / code scanning failures ("CodeQL")**
   - Retrieve the SARIF results or workflow logs from the failed run
   - Identify which CodeQL query/rule triggered the failure
   - Create one issue per distinct security finding:
     - Labels: `security`
     - Body includes: rule ID, severity, affected file(s) + line(s), explanation, suggested remediation

3c. **PyPI publish failures ("dynamic")**
   - Retrieve the workflow logs to identify the publish error
   - Common causes: version conflict, auth failure, build error, missing metadata
   - Create one issue:
     - Labels: `publish`, `packaging`
     - Body includes: full error from the publish step, likely cause, fix steps

4. **All issue types — common fields**
   - Title starts with `[copilot-auto] `
   - Body always includes:
     - Triggering commit SHA + link to the failed workflow run
     - The source workflow that failed
     - The source file(s) likely causing the failure
     - Suggested fix with code snippet
   - Base labels: `copilot`, `auto-generated`, `bug`
   - **Assign to** the pusher/triggerer (`${{ github.actor }}`)
   - Add this pinned comment at the top of the issue:
     > @copilot Please start working on this immediately. Open a draft PR with the fix when ready.

5. **If the fix is straightforward**
   - Instruct Copilot to implement it and open a linked **draft PR** to the issue.

6. **Summary**
   - If the workflow actually passed (guard) → log "All clear" and exit
   - Always link back to the triggering workflow run: `${{ github.event.workflow_run.html_url }}`

