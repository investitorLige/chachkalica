import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from fleet.models import FleetSettings
from videos import jobs
from videos.admin import VideoAddForm
from videos.models import Video
from videos.services.videos import list_video_files


class VideosBase(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fs = FleetSettings.load()
        fs.videos_dir = str(self.root)  # absolute -> used verbatim by videos_root()
        fs.save()

    def tearDown(self):
        self.tmp.cleanup()

    def _touch(self, name: str):
        (self.root / name).write_bytes(b"fake-mp4")


class ScannerTests(VideosBase):
    def test_lists_only_video_extensions_sorted(self):
        for name in ["b.mp4", "a.mov", "notes.txt", ".hidden.mp4", "c.webm"]:
            self._touch(name)
        self.assertEqual(list_video_files(), ["a.mov", "b.mp4", "c.webm"])

    def test_missing_root_is_empty(self):
        self.tmp.cleanup()  # remove the dir
        self.assertEqual(list_video_files(), [])


class AddFormTests(VideosBase):
    def test_requires_exactly_one_source(self):
        self._touch("clip.mp4")
        # neither
        self.assertFalse(VideoAddForm(data={"name": ""}).is_valid())
        # both
        both = VideoAddForm(data={"import_file": "clip.mp4",
                                  "source_url": "https://youtu.be/x"})
        self.assertFalse(both.is_valid())

    def test_import_defaults_name_to_stem(self):
        self._touch("clip.mp4")
        form = VideoAddForm(data={"import_file": "clip.mp4"})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["name"], "clip")

    def test_registered_files_excluded_from_choices(self):
        self._touch("clip.mp4")
        self._touch("other.mp4")
        Video.objects.create(name="clip", filename="clip.mp4", status=Video.READY)
        values = [v for v, _ in VideoAddForm().fields["import_file"].choices]
        self.assertIn("other.mp4", values)
        self.assertNotIn("clip.mp4", values)


class DownloadJobTests(VideosBase):
    def test_success_sets_filename_name_ready(self):
        video = Video.objects.create(
            name="https://youtu.be/x", source_url="https://youtu.be/x",
            status=Video.DOWNLOADING,
        )
        produced = self.root / "My Clip.mp4"
        produced.write_bytes(b"data")
        with mock.patch("videos.jobs.downloader.download", return_value=produced) as dl:
            jobs.download_video(video.id)
        dl.assert_called_once()
        video.refresh_from_db()
        self.assertEqual(video.status, Video.READY)
        self.assertEqual(video.filename, "My Clip.mp4")
        self.assertEqual(video.name, "My Clip")

    def test_failure_records_error(self):
        video = Video.objects.create(
            name="https://youtu.be/bad", source_url="https://youtu.be/bad",
            status=Video.DOWNLOADING,
        )
        with mock.patch("videos.jobs.downloader.download",
                        side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                jobs.download_video(video.id)
        video.refresh_from_db()
        self.assertEqual(video.status, Video.ERROR)
        self.assertIn("boom", video.last_error)

    def test_unique_name_dedupes(self):
        Video.objects.create(name="My Clip", filename="a.mp4", status=Video.READY)
        self.assertEqual(jobs._unique_name("My Clip"), "My Clip (2)")


class AdminRenderTests(VideosBase):
    def setUp(self):
        super().setUp()
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")

    def test_index_shows_videos_section(self):
        resp = self.client.get(reverse("admin:index"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Videos")

    def test_add_page_has_both_sources(self):
        (self.root / "clip.mp4").write_bytes(b"x")
        resp = self.client.get(reverse("admin:videos_video_add"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "import_file")
        self.assertContains(resp, "source_url")
        self.assertContains(resp, "id_quality")

    def test_change_page_embeds_player(self):
        (self.root / "clip.mp4").write_bytes(b"x")
        v = Video.objects.create(name="clip", filename="clip.mp4", status=Video.READY)
        resp = self.client.get(reverse("admin:videos_video_change", args=[v.pk]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "<video")

    def test_play_page_renders(self):
        (self.root / "clip.mp4").write_bytes(b"x")
        v = Video.objects.create(name="clip", filename="clip.mp4", status=Video.READY)
        url = reverse("admin:videos_video_play") + f"?video={v.pk}"
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "<video")


class StreamViewTests(VideosBase):
    def setUp(self):
        super().setUp()
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")
        (self.root / "clip.mp4").write_bytes(b"0123456789")
        self.video = Video.objects.create(
            name="clip", filename="clip.mp4", status=Video.READY,
        )

    def test_full_response_advertises_ranges(self):
        url = reverse("admin:videos_video_stream") + f"?video={self.video.pk}"
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Accept-Ranges"], "bytes")

    def test_range_request_returns_206_slice(self):
        url = reverse("admin:videos_video_stream") + f"?video={self.video.pk}"
        resp = self.client.get(url, headers={"range": "bytes=2-5"})
        self.assertEqual(resp.status_code, 206)
        self.assertEqual(resp["Content-Range"], "bytes 2-5/10")
        self.assertEqual(resp["Content-Length"], "4")
        self.assertEqual(b"".join(resp.streaming_content), b"2345")
