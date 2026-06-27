# Operations Manual

The end-to-end runbook for the annotator fleet: how to stand it up, annotate,
export to COCO, and recover when containers go down.

The fleet is a **Django app**. You drive it two interchangeable ways:

- the **admin panel** (`/admin/`) — point-and-click provisioning, setup, sync;
- **management commands** (`manage.py fleet_*`) — the same operations, scriptable.

Both call the same service layer, so anything you can do in one you can do in the
other. All Python commands use the module-local virtualenv:

```bash
.venv/bin/python manage.py <command>
```

## Architecture in one picture

```
 2+ Label Studio containers          Django app (web + worker)        stores
 (one per annotator, isolated)
 ┌───────────────────┐
 │ label-studio-bob  │  ANNOTATION_*  ┌──────────────────┐   Postgres ── fleet state
 │  :8081  project    ├──────────────►│  /hook  (web :9000)│      (ports, tokens, datasets,
 │  "x — bob"         │   webhook POST │  writes one .txt   │       projects)
 └───────────────────┘                │  per image, routed │
 ┌───────────────────┐                │  by ?annotator=…   │   Redis ──── job queue (RQ)
 │ label-studio-alice│  ANNOTATION_*  │                    │
 │  :8082  project    ├──────────────►│  rqworker runs the │   data/source/<dataset>/ (shared, read)
 │  "x — alice"       │   webhook POST │  provision/sync    │      *.jpg + classes.txt
 └───────────────────┘                │  jobs              │   data/target/<dataset>/ (shared, write)
                                       └──────────────────┘      bob/*.txt, bob.coco.json …
```

- One **container per annotator** (isolated DB, volume, token, port).
- One **project per dataset** inside each container, titled `"<dataset> — <username>"`.
- All containers read the **same** `data/source`; all write the **same**
  `data/target`, namespaced per annotator so they never collide.
- Neither LS instance writes labels directly — both POST to the Django
  `/hook` endpoint, which writes the files. (This replaced the old standalone
  `coco_sync` Flask service; it now lives inside the app and reads tokens from
  the database instead of a YAML file.)
- Long operations (provisioning a container ~90s, syncing a project) run as
  **background jobs** on the RQ worker, so the admin/CLI never blocks.

## What persists when a container dies

Recovery is painless as long as these survive:

| Survives | Where | Holds |
|---|---|---|
| Fleet state | **Postgres** (Docker volume `fleet-postgres`) | annotators, ports, passwords, tokens, datasets, projects |
| LS data | Docker volume `label-studio-<user>-data` | projects, tasks, **annotations, registered webhooks**, legacy-token setting |
| Images / labels | `data/source`, `data/target` (host) | source images, exported txts + COCO |

**As long as the Postgres volume and the LS volumes exist, you lose nothing.**
(Redis holds only transient job state — losing it just drops in-flight jobs.)

---

## First-time setup

### 0. Prerequisites

- Docker + `docker compose` available.
- A dataset at `data/source/<dataset>/` containing a `classes.txt` (one class
  per non-empty line; an optional `# tools: bbox, polygon, keypoint` directive)
  plus the images.

### 1. Bring up Postgres + Redis, migrate, create an admin user

```bash
docker compose up -d postgres redis
.venv/bin/python manage.py migrate
.venv/bin/python manage.py createsuperuser
```

### 2. Run the web process and a worker

The **recommended** path runs everything in compose (web on host port 9000,
worker with the docker socket mounted so it can start containers):

```bash
docker compose up -d           # postgres, redis, web (:9000), worker
curl -s localhost:9000/health  # -> {"ok": true}
```

For **local development** run them on the host instead:

```bash
# the webhook + admin
FLEET_LS_HOST=localhost .venv/bin/python manage.py runserver 0.0.0.0:9000
# in a second shell, the job worker
FLEET_LS_HOST=localhost .venv/bin/python manage.py rqworker default
```

> `FLEET_LS_HOST` is how the webhook/worker reach the LS containers. In compose
> it is `host.docker.internal` (the default); running on the host directly, set
> it to `localhost`. The worker **must** have access to the Docker socket and
> the `docker` CLI — compose mounts it for you.

> **Tip:** to run a one-off operation synchronously without a worker, set
> `RQ_ASYNC=False` — jobs execute inline. The `manage.py fleet_*` commands
> already run synchronously regardless (they call the service layer directly).

You will use these once at the top, then mostly the admin.

---

## Full flow — from scratch to exported COCO

Each step shows the **admin** way and the **CLI** way.

### 1. Provision annotator containers

**Admin:** *Fleet → Annotators → Add*, type a username, save (this reserves the
next free port from 8081 and generates a password + token). Then select the
row(s) → action **“Provision / restore selected annotators.”**

**CLI:**

```bash
.venv/bin/python manage.py fleet_add bob
.venv/bin/python manage.py fleet_add alice
.venv/bin/python manage.py fleet_add            # provision every annotator row
```

Provisioning starts the container **with `--add-host=host.docker.internal:host-gateway`**
(required for live webhooks on Linux), waits for first boot, enables legacy
token auth, and resolves a working API token. In the admin the row's status
flows `queued → running → ok` (refresh to watch it); on the CLI it runs inline.

> Provisioning is **idempotent**: a running container with a working token is a
> no-op, and a **stopped** container is simply started. Re-running is always safe.

### 2. Create the dataset project in each container + register webhooks

**Admin:** *Fleet → Datasets → Add*, set the name to the directory under
`data/source/` (e.g. `dataset1`), save. Select it → action **“Set up selected
dataset(s) for all active annotators.”**

**CLI:**

```bash
.venv/bin/python manage.py fleet_setup_dataset <dataset> --all
# or a single annotator:
.venv/bin/python manage.py fleet_setup_dataset <dataset> --annotator bob
```

For each running container this:

1. validates `data/source/<dataset>/classes.txt` exists,
2. creates the project `"<dataset> — <username>"` (idempotent — skips if it
   already exists),
3. attaches a local-files storage at `source/<dataset>` and imports one task
   per image,
4. registers the webhook
   `http://host.docker.internal:9000/hook?annotator=<u>&dataset=<d>&project_id=<id>`
   for `ANNOTATION_CREATED/UPDATED` + `ANNOTATIONS_DELETED`,
5. saves a **Project** row remembering the real `ls_project_id` + `webhook_id`.

The webhook base URL comes from *Fleet settings → webhook_url* (default
`http://host.docker.internal:9000`).

### 3. Verify

```bash
.venv/bin/python manage.py fleet_list   # each annotator [running]
```

Open each annotator's URL (e.g. `http://localhost:8081`) and log in with the
email/password from the Annotator row. (Secrets are stored in the database but
hidden in the admin UI; read them with `manage.py shell` if you need them.)

### 4. Annotate → labels export live

As annotators submit/update/delete in the UI, `/hook` writes one `.txt` per
image under `data/target/<dataset>/<username>/`. Header `W H`, then one line per
object: `class_id cx cy w h` (+ normalized polygon points). Only **bbox +
polygon** are exported (brush / keypoint regions are ignored).

### 5. Build / refresh the COCO files

The live webhook keeps the per-image txts current but does **not** touch COCO.
Compile (and reconcile) with sync.

**Admin:** *Datasets* → action **“Sync all projects of selected dataset(s),”**
or *Projects* → select rows → **“Sync selected projects.”**

**CLI:**

```bash
.venv/bin/python manage.py fleet_sync                       # whole fleet, every dataset
.venv/bin/python manage.py fleet_sync <dataset> --all       # one dataset, whole fleet
.venv/bin/python manage.py fleet_sync <dataset> --annotator bob
```

`sync` *pulls* authoritative state from Label Studio (by the stored
`ls_project_id`), rebuilds every per-image txt, prunes stale files, and writes
`data/target/<dataset>/<username>.coco.json`. It does not depend on the webhook
chain, so it is also the catch-all for backfilling anything the live webhook
missed.

---

## Migrating from the old YAML tooling

If you are coming from the pre-Django `fleet.py` (state in
`configs/fleet.local.yaml` + `configs/register.yaml`), import it once:

```bash
.venv/bin/python manage.py import_fleet_yaml
# also recover Project rows (their ls_project_id) by querying live containers:
.venv/bin/python manage.py import_fleet_yaml --backfill-projects
```

This upserts Fleet settings + one Annotator row per username (in the live fleet
→ `active`; only in the register → `retired`), preserving every port/password/
token. It is idempotent. The old `fleet.py` / `label-studio.py` are kept under
`legacy/` for reference; the standalone `coco_sync/` service is superseded by
the `/hook` view and no longer needs to run.

---

## Recovery — containers went down

> First check what state they are in: `docker ps -a --filter name=label-studio-`

The Django **web + worker** carry `restart: unless-stopped` in compose and
self-heal on a Docker restart. The **LS containers have no restart policy**, so
they stay `Exited` after a reboot until started.

### Case A — containers stopped but still exist (reboot / crash / `docker stop`)

Re-provision them — provisioning now starts a stopped container for you:

```bash
docker compose up -d                       # ensure web + worker are up
.venv/bin/python manage.py fleet_add        # starts every stopped container
.venv/bin/python manage.py fleet_list       # confirm [running]
curl -s localhost:9000/health
```

(Or in the admin: select the annotators → **Provision / restore**.)

> A *legacy* container created without `--add-host` stays without it (live
> webhooks still broken) — recreate it via Case B to pick up the host-gateway.

### Case B — containers removed (`docker rm`) but volumes intact

Re-provision; this rebuilds the container and re-attaches the host-gateway:

```bash
.venv/bin/python manage.py fleet_add bob    # recreates with SAME volume/port/token + --add-host
.venv/bin/python manage.py fleet_add        # or recreate everything
```

Because the Annotator rows still exist in Postgres, provisioning reuses each
annotator's password/token/volume/port and just rebuilds the container.
**Projects and webhooks come back from the volume DB — no setup-dataset
needed** (it is idempotent if you want to re-verify/re-register).

### Case C — volumes also deleted (real data loss)

LS-side projects and annotations are **gone and not auto-recoverable**. Only the
exported labels in `data/target/…` survive on disk, and there is no built-in
import-back. To rebuild empty projects:

```bash
.venv/bin/python manage.py fleet_add                       # recreate containers
.venv/bin/python manage.py fleet_setup_dataset <dataset> --all   # recreate projects, re-import, re-register
```

Past annotations must be redone. **Protect the volumes — they are the source of
truth.**

### Always after recovery — backfill missed deliveries

If annotators worked while the web process was down (or a container lacked the
host-gateway), those live deliveries were lost. Reconcile from LS:

```bash
.venv/bin/python manage.py fleet_sync
```

---

## When the live webhook won't fire (and why)

The chain is: LS event → LS POSTs → network → `/hook` → file write. It can break
at any link.

| Symptom / cause | Affects | Notes |
|---|---|---|
| Webhook not registered (project not made by setup-dataset) | all | nothing fires; no log anywhere |
| Saved a **draft**, not submitted | all | drafts are not annotations |
| LS container lacks `--add-host` (legacy container) | all | POST can't resolve `host.docker.internal`; **no web log** |
| Web process down / port 9000 unreachable | all | connection refused |
| `/hook` can't reach LS API back (bad/rotated token, legacy auth off, wrong `FLEET_LS_HOST`) | **delete only** (create/update ride the inlined payload) | 500 in web log |
| `classes.txt` missing for the dataset | all | 500 in web log |
| Annotator row missing or `retired` | that annotator | `/hook` returns 404 |
| Region is brush / keypoint, or label not in `classes.txt` | that region | silently dropped (a `log.warning` is emitted) |
| `data/target` read-only or wrong uid | all writes | check the volume mount / process uid |

The two genuinely **silent** failures (no log anywhere) are *not registered* and
*POST can't leave the LS container*. Everything else leaves a trace. Background
**job** failures (provision/sync) are visible at `/django-rq/` (the failed-job
registry) and in each row's `last_status` / `last_error`. When in doubt,
`fleet_sync` bypasses the whole push chain and is the ground truth.

---

## Cleanup

**Admin:** *Annotators* → **“Remove containers (keep volume + row)”** (restorable
later) or **“Purge (delete container, volume, AND row)”** (irreversible).

**CLI:**

```bash
.venv/bin/python manage.py fleet_remove bob            # stop + remove container; keep volume + row (status → retired)
.venv/bin/python manage.py fleet_remove bob --purge    # also delete the volume and the row
docker compose down                                    # stop the whole stack (add -v to drop Postgres too)
```
