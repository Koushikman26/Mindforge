#!/usr/bin/env bash
set -euo pipefail
# Usage: IMAGE_TAG=<tag> ./scripts/deploy.sh prod
ENV=${1:-prod}
COMPOSE_FILE="infra/docker-compose.prod.yml"

echo "Deploying to $ENV with IMAGE_TAG=${IMAGE_TAG:-latest}"
docker compose -f $COMPOSE_FILE pull
docker compose -f $COMPOSE_FILE up -d
docker compose -f $COMPOSE_FILE ps
