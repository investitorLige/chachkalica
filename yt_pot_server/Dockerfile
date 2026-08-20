# Thin wrapper around brainicism/bgutil-ytdlp-pot-provider (see docker-compose.yml
# yt-pot-server + chachkalica/videos/services/downloader.py).
#
# The published image ships with NO CA certificates at all (no /etc/ssl/certs) —
# every outbound HTTPS call it makes to fetch YouTube's BotGuard challenge script
# fails with "unable to get local issuer certificate", which surfaces to yt-dlp
# as a plain 500 from POST /get_pot and silently falls back to an unauthorized
# download that YouTube then 403s. Installing ca-certificates is the whole fix.
FROM brainicism/bgutil-ytdlp-pot-provider

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
USER node
