# Docker + CI for SafeStride

This guide covers running the FastAPI service and Next.js app with Docker Compose, and the CI workflow.

## Files added
- **Dockerfile.api**: Python FastAPI (CPU-only), runs `uvicorn src.service.app:app`.
- **Dockerfile.app**: Next.js app. Multi-stage build for production (`runner`) and an optional `dev` target.
- **docker-compose.yml**: Spins up `api` and `app` together with a shared `.env`.
- **.github/workflows/ci.yml**: GitHub Actions running Ruff lint, pytest, and building the images.
- **requirements.api.txt** and **requirements.dev.txt**: Python deps for API and for dev/tests.
- **.env.example**: Example environment variables used by both services.

## Prerequisites
- Docker Desktop (Docker Compose v2 included)
- Make a copy of the example env file

```bash
cp .env.example .env
# Edit the values as needed
```

Minimal variables:
- `SERVICE_TOKEN` – required to call `/predict` (used by `src/service/security.py`). Not needed for `/health`.
- `NEXT_PUBLIC_API_BASE_URL` – base URL for the API (e.g., `http://api:8000` inside Compose, `http://localhost:8000` on host).

## Run locally with Docker Compose

```bash
docker compose up --build
```

- API: http://localhost:8000/health should return `{ "ok": true }`.
- App: http://localhost:3000 should load the Next.js site.

To rebuild only one service:
```bash
docker compose build api
# or
docker compose build app
```

To run the app in dev target (optional):
```bash
docker build --target dev -f Dockerfile.app -t safestride-app:dev .
```
Then run with:
```bash
docker run -p 3000:3000 safestride-app:dev
```

## CI details
Workflow: `.github/workflows/ci.yml`
- **Lint**: `ruff check .`
- **Tests**: `pytest -q` (installing `requirements.dev.txt`)
- **Build images**: builds both `Dockerfile.api` and `Dockerfile.app` to ensure Dockerfiles stay valid.

## Notes
- The API looks for models under `models_registry/<subject>/<trial>/model.pkl` unless `model_path` is provided. Not needed for `/health`.
- If you need the frontend to call the API from the browser, ensure `NEXT_PUBLIC_API_BASE_URL` is correctly set during build/run.
