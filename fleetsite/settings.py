"""
Django settings for fleetsite — the Label Studio annotator-fleet manager.

Configuration is environment-driven so the web process, the rq worker, and the
webhook receiver can all share one Postgres + Redis instance. Defaults match
the bundled docker-compose services, so a local checkout runs with no env vars
once `docker compose up -d postgres redis` is running.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load a local .env (if present) into the environment before reading settings.
# Real values for env vars below override these; in containers, compose's
# `environment:` takes precedence over anything here.
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# SECURITY WARNING: set DJANGO_SECRET_KEY in production.
SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "django-insecure-f0laaiqr$0%ddyayg_0vl+6^_95v*bl)k1_3$m(c37z88@2=h*",
)

DEBUG = _env_bool("DJANGO_DEBUG", True)

# host.docker.internal is included by default: Label Studio containers POST
# annotation webhooks to http://host.docker.internal:9000/hook, so that Host
# header must be trusted or the /hook receiver 400s with DisallowedHost.
ALLOWED_HOSTS = os.getenv(
    "DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,0.0.0.0,host.docker.internal"
).split(",")

# The webhook is hit by Label Studio containers via the docker host gateway, so
# its POSTs arrive with a Host header we must trust for CSRF-exempt POSTs.
CSRF_TRUSTED_ORIGINS = [
    o for o in os.getenv("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",") if o
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_rq",
    "fleet",
    "training",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # Serves static files (admin CSS + the fleet theme) under gunicorn, where
    # Django would not on its own. Must sit right after SecurityMiddleware.
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "fleetsite.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # Project templates win over app templates, so this overrides the
        # admin's own base_site.html (admin is listed before fleet in apps).
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "fleetsite.wsgi.application"


# Database — Postgres. The webhook + provisioning both write, and the rq worker
# is a separate process; Postgres handles the concurrent writers cleanly.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("POSTGRES_DB", "fleet"),
        "USER": os.getenv("POSTGRES_USER", "fleet"),
        "PASSWORD": os.getenv("POSTGRES_PASSWORD", "fleet"),
        "HOST": os.getenv("POSTGRES_HOST", "127.0.0.1"),
        "PORT": os.getenv("POSTGRES_PORT", "5432"),
    }
}


# Redis / RQ — the queue that runs the long provisioning + sync jobs.
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

# RQ_ASYNC=False runs jobs inline (no worker/Redis round-trip) — handy for tests
# and quick local runs.
RQ_ASYNC = _env_bool("RQ_ASYNC", True)

RQ_QUEUES = {
    # First boot runs DB migrations (wait_until_http 180s) then resolves a token
    # (90s); give the job comfortably more than 270s before the worker kills it.
    "default": {"URL": REDIS_URL, "DEFAULT_TIMEOUT": 900, "ASYNC": RQ_ASYNC},
}

RQ_SHOW_ADMIN_LINK = True


AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# WhiteNoise compresses + serves collected static files from STATIC_ROOT.
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# ---------------------------------------------------------------------------
# Fleet-specific settings
# ---------------------------------------------------------------------------
# Label Studio fleet state records each instance as http://localhost:<port>,
# which is correct from the host. The webhook receiver, however, reaches the
# instances via the docker host gateway, so it rewrites localhost -> LS_HOST.
LS_HOST = os.getenv("FLEET_LS_HOST", "host.docker.internal")

# When the worker runs inside a container, `docker run -v` mount sources must be
# HOST paths, not the worker's /app/data. Set this to the host's ./data dir
# (compose passes ${PWD}/data). Empty when running on the host — the resolved
# path is already correct there.
FLEET_HOST_DATA_DIR = os.getenv("FLEET_HOST_DATA_DIR", "")
