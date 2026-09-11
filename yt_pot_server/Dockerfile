# Thin wrapper around brainicism/bgutil-ytdlp-pot-provider (see docker-compose.yml
# yt-pot-server + chachkalica/videos/services/downloader.py).
#
# The published image ships with NO CA certificates at all (no /etc/ssl/certs) —
# every outbound HTTPS call it makes to fetch YouTube's BotGuard challenge script
# fails with "unable to get local issuer certificate", which surfaces to yt-dlp
# as a plain 500 from POST /get_pot and silently falls back to an unauthorized
# download that YouTube then 403s. Installing ca-certificates is the whole fix.
#
# Bootstrapping it on this network is circular: apt over plain HTTP is mangled by
# the TLS-inspecting proxy (responses arrive with no status line, which apt
# reports as "Bad header line Bad header data"), and apt over HTTPS needs the very
# CA bundle we are here to install. So seed a bundle from alpine, which ships one
# in the base image, and only then install the real Debian package.
FROM alpine:3 AS certs

FROM brainicism/bgutil-ytdlp-pot-provider

USER root
COPY --from=certs /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt

# deb.debian.org is TLS-intercepted on this network: plain HTTP comes back with
# no status line and HTTPS is MITM'd with a self-signed "atlas-MASTER" cert, so
# apt can use neither. cloudfront.debian.net is an official Debian CDN the proxy
# leaves alone.
# Override with --build-arg DEBIAN_MIRROR=https://deb.debian.org off this network.
ARG DEBIAN_MIRROR=https://cloudfront.debian.net
RUN sed -i -E "s|https?://deb\.debian\.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
USER node
