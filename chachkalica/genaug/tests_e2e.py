"""End to end against a real, running genaug-backend (mock editor, no GPU).

Skipped unless ``GENAUG_E2E=1``. Unlike ``genaug.tests``, nothing is faked:
the worker-side code calls the backend over HTTP, the backend reads and writes
real files on the shared ``data/`` mount, and the build lands as a dataset.
The temp dataset has to live *under* that mount (the backend refuses paths
outside it), so it is created in ``data/genaug/_e2e/`` and removed afterwards.

    docker compose up -d genaug-backend
    docker compose run --rm -T -v "$PWD/chachkalica:/app" -e GENAUG_E2E=1 web \\
        python manage.py test genaug.tests_e2e --keepdb --noinput
"""

import json
import os
import shutil
import unittest
import uuid
from pathlib import Path

import cv2
from django.conf import settings
from django.test import TestCase

from fleet.models import Dataset, FleetSettings
from genaug import jobs
from genaug.models import AugBuild, AugSession, Status
from genaug.services import backend, builder, previews
from genaug.tests import BOX, draw_object, textured, to_yolo


@unittest.skipUnless(os.environ.get("GENAUG_E2E"), "set GENAUG_E2E=1 with genaug-backend running")
class LiveBackendTests(TestCase):
    def setUp(self):
        health = backend.health()
        self.assertEqual(health["status"], "ok")
        self.scratch = Path(settings.BASE_DIR) / "data" / "genaug" / "_e2e" / uuid.uuid4().hex
        self.addCleanup(shutil.rmtree, self.scratch, ignore_errors=True)
        root = self.scratch / "source"
        ds = root / "e2e_ppe"
        (ds / "images").mkdir(parents=True)
        (ds / "labels").mkdir()
        (ds / "classes.txt").write_text("helmet\nvest\n")
        for i in range(4):
            cv2.imwrite(str(ds / "images" / f"f{i}.jpg"), draw_object(textured(i), BOX))
            (ds / "labels" / f"f{i}.txt").write_text(to_yolo(BOX, i % 2))
        fs = FleetSettings.load()
        fs.source_dir = str(root)
        fs.save()
        self.root = root
        self.dataset = Dataset.objects.create(name="e2e_ppe", has_labels=True)
        # Previews and builds write to the shared cache; keep it inside scratch.
        os.environ["GENAUG_DATA_DIR"] = str(self.scratch / "genaug")
        self.addCleanup(os.environ.pop, "GENAUG_DATA_DIR", None)

    def test_previews_then_build_through_the_real_backend(self):
        session = AugSession.objects.create(name="e2e", dataset=self.dataset, editor="mock",
                                            preview_count=2)
        session.preview_images = previews.pick_preview_images(session)
        session.save()
        prompt = previews.create_prompt(session, name="night",
                                        text="night cctv with sensor noise")
        jobs.run_prompt_previews(prompt.pk)
        prompt.refresh_from_db()
        self.assertEqual(prompt.status, Status.OK, prompt.last_error)
        for pv in prompt.previews.all():
            self.assertTrue((Path(os.environ["GENAUG_DATA_DIR"]) / pv.output_relpath).is_file())
            self.assertIsNotNone(pv.accepted)

        build = AugBuild.objects.create(
            session=session, source_dataset=self.dataset,
            source_labels_dir=str(self.root / "e2e_ppe" / "labels"),
            output_name="e2e_ppe__genaug", prompts_snapshot=[prompt.generation()],
            fraction=1.0, variants_per_image=1, editor="mock",
        )
        jobs.run_build(build.pk)
        build.refresh_from_db()
        self.assertEqual(build.status, Status.OK, build.last_error)
        self.assertEqual(build.variants_attempted, 4)
        # The two previewed images come back from the cache, not the backend.
        self.assertEqual(build.variants_cached, 2)
        out = self.root / "e2e_ppe__genaug"
        records = [json.loads(l) for l in (out / builder.MANIFEST).read_text().splitlines()]
        self.assertEqual(len(records), 4)
        accepted = [r for r in records if r["accepted"]]
        for record in accepted:
            self.assertTrue((out / "images" / record["generated"]).is_file())
        print("\n" + jobs.summary(build))
