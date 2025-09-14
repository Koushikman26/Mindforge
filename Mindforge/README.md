# Team Development Environment

This repository bootstraps a complete development environment with:
- Version control and Git branching strategy
- Linux-based dev servers (Dockerized services)
- Containerization via Docker & Compose
- CI/CD pipeline (GitHub Actions) with build/test/lint and deploy jobs
- Three environments: **development**, **staging**, **production**
- Access provisioning checklist
- "Hello World" application to prove deployment

## Quick start (local dev)
```bash
# 1) Clone and bootstrap
git clone <YOUR-REPO-URL> team-dev-env
cd team-dev-env
cp .env.example .env

# 2) Start services
docker compose -f infra/docker-compose.dev.yml up --build

# 3) Visit the app
# Backend: http://localhost:8000
# Health:  http://localhost:8000/health
```

## Branching strategy (GitHub Flow with env gates)
- `main`: production
- `staging`: staging
- feature branches: `feat/<short-desc>`
- fix branches: `fix/<short-desc>`

**Rules**
- Feature → PR into `staging` (CI must pass)
- Release: `staging` → `main` via PR (tagged)
- Hotfix: `fix/*` → `main` with back-merge into `staging`

See `docs/BRANCHING.md` for details.
