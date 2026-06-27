#!/usr/bin/env sh
set -eu

checkpoint="${SAM_CHECKPOINT:-/models/sam_vit_b_01ec64.pth}"
model_type="${SAM_MODEL_TYPE:-vit_b}"

if [ ! -f "$checkpoint" ]; then
  mkdir -p "$(dirname "$checkpoint")"

  case "$model_type" in
    vit_h)
      url="${SAM_CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth}"
      ;;
    vit_l)
      url="${SAM_CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth}"
      ;;
    vit_b)
      url="${SAM_CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth}"
      ;;
    *)
      if [ -z "${SAM_CHECKPOINT_URL:-}" ]; then
        echo "SAM checkpoint is missing: $checkpoint" >&2
        echo "Set SAM_CHECKPOINT_URL for SAM_MODEL_TYPE=$model_type." >&2
        exit 1
      fi
      url="$SAM_CHECKPOINT_URL"
      ;;
  esac

  tmp="${checkpoint}.download"
  echo "Downloading SAM checkpoint for $model_type to $checkpoint"
  curl -L --fail --retry 3 --output "$tmp" "$url"
  mv "$tmp" "$checkpoint"
fi

exec label-studio-ml start ml_backends/sam --host 0.0.0.0 --port 9090
