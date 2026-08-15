import logging
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_CHECKPOINTS_YAML = Path(__file__).parent.parent / "configs" / "checkpoints.yaml"
_CACHE_DIR = Path(__file__).parent.parent / "checkpoint_cache"
_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _load_aliases() -> dict:
    if not _CHECKPOINTS_YAML.exists():
        return {}
    with open(_CHECKPOINTS_YAML) as f:
        data = yaml.safe_load(f)
    return data.get("aliases", {}) if data else {}


def _interpolate(s: str) -> str:
    def _replace(m):
        var = m.group(1)
        val = os.environ.get(var)
        if val is None:
            raise ValueError(f"${{{var}}} not set")
        return val
    return _VAR_RE.sub(_replace, s)


def resolve_checkpoint(uri: str) -> str:
    """
    Resolve a checkpoint identifier to a local path.

    Accepts:
      - Named alias (e.g. "base") — looked up in configs/checkpoints.yaml
      - gs:// URI — downloaded to ./checkpoint_cache/ and local path returned
      - Local path — returned unchanged
    """
    aliases = _load_aliases()
    if uri in aliases:
        uri = _interpolate(aliases[uri])

    if uri.startswith("gs://"):
        return _download_gcs(uri)

    return uri


def _gcs_local_path(uri: str) -> Path:
    """Map gs://bucket/prefix → checkpoint_cache/bucket/prefix."""
    return _CACHE_DIR / uri[5:]  # strip "gs://"


def _download_gcs(uri: str) -> str:
    from google.cloud import storage  # lazy import — only available on the VM

    local_root = _gcs_local_path(uri)
    without_scheme = uri[5:]
    bucket_name, _, blob_prefix = without_scheme.partition("/")

    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=blob_prefix))

    if not blobs:
        raise FileNotFoundError(f"No objects found at {uri}")

    logger.info("Downloading %d object(s) from %s → %s", len(blobs), uri, local_root)

    for blob in blobs:
        if blob.name.endswith("/"):  # skip GCS directory placeholder objects
            continue
        rel = blob.name[len(blob_prefix):].lstrip("/")
        dest = local_root / rel if rel else local_root
        if dest.exists():
            logger.debug("Skipping cached %s", dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("  %s", blob.name)
        blob.download_to_filename(str(dest))

    return str(local_root)
