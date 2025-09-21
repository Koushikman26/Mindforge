# Branching & Release Strategy

We use a lightweight GitHub Flow with environment gates.

## Branch types
- `main`: production-ready, protected branch
- `staging`: integration branch for UAT
- `feat/*`, `fix/*`, `chore/*`: short-lived branches

## Pull Requests
- All PRs require successful CI (lint, test, build)
- Require at least 1 code review approval
- Enforce conventional commits in PR titles (optional)

## Releases
- Promote from `staging` → `main` by PR
- Tag releases: `vX.Y.Z` (semantic versioning)
- CI creates a Release on tag push
