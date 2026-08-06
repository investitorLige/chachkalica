"""Shared-token auth for the build node.

One token per node, held in ``BUILDNODE_TOKEN`` here and on the ``BuildNode`` row
in Django. This is the same threat model the rest of the stack already documents
(``chachkalica/fleet/models.py``): a plaintext secret on a trusted network, not a
credential system.

There is no anonymous mode. A build node accepts arbitrary ONNX graphs and runs
them through a compiler on a GPU; "nobody will find the port" is not an access
policy. A node with no token configured refuses every request, including
``/health`` — an unauthenticated capability probe is free reconnaissance, and
answering it tells a scanner exactly which GPU it just found.
"""

import os
import secrets

from fastapi import Header, HTTPException

_ENV_VAR = "BUILDNODE_TOKEN"


def configured_token() -> str:
    return (os.environ.get(_ENV_VAR) or "").strip()


def require_token(authorization: str = Header(default="")) -> None:
    """FastAPI dependency: reject anything without the node's bearer token."""
    expected = configured_token()
    if not expected:
        # 503, not 401: the caller's credentials aren't the problem, the node is
        # misconfigured. A 401 would send an operator hunting for a bad token.
        raise HTTPException(
            status_code=503,
            detail=f"This build node has no {_ENV_VAR} set and will not serve requests.",
        )

    scheme, _, presented = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not presented:
        raise HTTPException(
            status_code=401, detail="Expected an 'Authorization: Bearer <token>' header."
        )
    # compare_digest, not ==, so a wrong token can't be recovered a byte at a time.
    if not secrets.compare_digest(presented.strip(), expected):
        raise HTTPException(status_code=401, detail="Bad build-node token.")
