# Environment Setup

## Prerequisites
- Git, Docker, Docker Compose v2
- Python 3.11+

## Local development
```bash
cp .env.example .env
docker compose -f infra/docker-compose.dev.yml up --build
```

## Staging/Production (server)
- Install Docker/Compose on a Linux VM
- Copy repo (or use CI to deploy)
- Create environment-specific `.env` (see examples)
- Run:
```bash
docker compose -f infra/docker-compose.prod.yml pull
docker compose -f infra/docker-compose.prod.yml up -d
```

## Access
- Add teammates to the repo with `Read` or `Write`
- Share server SSH access via team account or per-user keys
- Store secrets in GitHub Actions secrets (never commit them)
