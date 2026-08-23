#!/bin/sh
set -eu

if [ -z "${ANALYTICS_DATABASE_PASSWORD:-}" ]; then
    echo "ANALYTICS_DATABASE_PASSWORD is required" >&2
    exit 1
fi

psql \
    --set=ON_ERROR_STOP=1 \
    --set=analytics_password="$ANALYTICS_DATABASE_PASSWORD" \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" <<'SQL'
ALTER ROLE rag_ionis_analytics WITH PASSWORD :'analytics_password';
SQL
