"""The one interface every generative image editor implements.

``edit(image, prompt, seed, params) -> image`` is the whole contract between
this container and the rest of the system. FireRed is the first model behind
it; Qwen-Image-Edit, LongCat or anything else instruction-driven is one more
class and one registry row, with no change to the Django side, the dataset
builder or the validator.

Two promises every editor keeps, because the labels depend on them:

* **The output has the input's exact pixel size.** Editors work at whatever
  resolution their model likes, then resize back. Source boxes are copied onto
  the result unchanged, so a size change would silently misplace every one.
* **A given (image, prompt, seed, params) is deterministic** as far as the
  model allows, so the builder's cache key means what it says.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # PIL lives in the backend image only; Django imports this too.
    from PIL import Image


@dataclass
class EditParams:
    negative_prompt: str = " "
    num_inference_steps: int = 40
    true_cfg_scale: float = 4.0


class GenerativeImageEditor:
    """A loaded instruction-driven image editor."""

    #: Set by concrete editors so the service can report what is warm.
    label: str = "editor"

    def load(self) -> None:
        """Bring the model into memory. Called once, before the first edit."""
        raise NotImplementedError

    def edit(self, image: "Image.Image", prompt: str, seed: int, params: EditParams) -> "Image.Image":
        """Return ``image`` edited per ``prompt``, at ``image``'s exact size."""
        raise NotImplementedError

    def unload(self) -> None:
        """Release the model's memory. The default is enough for most editors."""

    def describe(self) -> dict:
        """What the service reports about the warm editor (device, dtype, …)."""
        return {"label": self.label}
