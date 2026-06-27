# chachkalica — Annotator Fleet Manager

A Django app for running an **image-annotation fleet** on top of [Label Studio](https://labelstud.io/).
It provisions one isolated Label Studio container per annotator, manages their
projects, captures annotations via webhook, and exports the result to COCO.

## What it does

- **One container per annotator** — isolated DB, volume, token, and port, so
  annotators never collide.
- **One project per dataset** inside each container, titled `"<dataset> — <username>"`.
- **Shared source / namespaced output** — all containers read the same
  `data/source/<dataset>/` (images + `classes.txt`) and write to
  `data/target/<dataset>/`, namespaced per annotator.
- **Webhook capture** — Label Studio POSTs annotations to the Django `/hook`
  endpoint, which writes one file per image (no LS instance writes labels directly).
- **Background jobs** — long operations (container provisioning ~90s, project
  sync) run on an [RQ](https://python-rq.org/) worker, so the admin UI / CLI never blocks.
- **SAM ML backend** — interactive point-click SAM and text-prompt Grounding-SAM
  preannotation through a single `/predict` endpoint (see `ml_backends/sam/`).
- **COCO export** — exported `.txt` labels are reconciled into per-annotator
  COCO JSON.

You drive it two interchangeable ways — the **admin panel** (`/admin/`) or
**management commands** (`manage.py fleet_*`). Both call the same service layer.

## Architecture

```
 2+ Label Studio containers          Django app (web + worker)        stores
 (one per annotator, isolated)
 ┌───────────────────┐
 │ label-studio-bob  │  webhook POST  ┌──────────────────┐   Postgres ── fleet state
 │  project "x — bob"├───────────────►│  /hook (web :9000)│      (ports, tokens, datasets)
 └───────────────────┘                │  writes one .txt  │
 ┌───────────────────┐                │  per image        │   Redis ──── RQ job queue
 │ label-studio-alice├───────────────►│                   │
 └───────────────────┘                └──────────────────┘   data/source, data/target (host)
```

## Tech stack

- **Django 5.2** (web + admin), **Postgres** (fleet state), **Redis + RQ** (jobs)
- **Docker / docker compose** for orchestration
- **Gunicorn**, **WhiteNoise** for serving

## Quick start

Prerequisites: Docker + `docker compose`, and a dataset at
`data/source/<dataset>/` containing a `classes.txt` plus images.

```bash
# 1. Configure environment
cp .env.example .env            # then set DJANGO_SECRET_KEY etc.

# 2. Bring up Postgres + Redis, migrate, create an admin user
docker compose up -d postgres redis
.venv/bin/python manage.py migrate
.venv/bin/python manage.py createsuperuser

# 3. Run the full stack (web on :9000, worker with docker socket mounted)
docker compose up -d
curl -s localhost:9000/health   # -> {"ok": true}
```

All Python commands use the module-local virtualenv (`.venv/bin/python`); do not
rely on system Python/pip.

## Documentation

- [Operations Manual](manual.md) — full end-to-end flow + recovery runbook
- [Configuration](docs/configuration.md)
- [Commands](docs/commands.md)
- [Annotator Fleet](docs/annotator-fleet.md)
- [Repository Agent Notes](AGENTS.md) — SAM backend design notes

## Repository layout

| Path | Purpose |
|---|---|
| `fleet/` | Core Django app — models, views, jobs, services, management commands |
| `fleetsite/` | Django project settings / WSGI / ASGI |
| `ml_backends/sam/` | Interactive SAM + Grounding-SAM ML backend |
| `coco_sync/` | COCO reconciliation / export |
| `configs/` | Label Studio instance + project configs |
| `docs/` | Configuration, commands, and fleet docs |
| `legacy/` | Superseded standalone scripts |
