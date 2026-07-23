from unittest import mock

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.urls import reverse

from cameras.admin import mask_rtsp_url
from cameras.models import Camera, validate_rtsp_url

RTSP_URL = "rtsp://admin:AxProVideo2024@10.10.10.24:554/Streaming/Channels/101"


class ValidatorTests(TestCase):
    def test_accepts_rtsp_scheme(self):
        validate_rtsp_url(RTSP_URL)  # no raise

    def test_rejects_non_rtsp_scheme(self):
        with self.assertRaises(ValidationError):
            validate_rtsp_url("https://example.com/stream")


class MaskUrlTests(TestCase):
    def test_masks_password(self):
        self.assertEqual(
            mask_rtsp_url(RTSP_URL),
            "rtsp://admin:••••@10.10.10.24:554/Streaming/Channels/101",
        )

    def test_leaves_credential_free_url_untouched(self):
        url = "rtsp://10.10.10.24:554/Streaming/Channels/101"
        self.assertEqual(mask_rtsp_url(url), url)


class SaveModelTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")

    def test_successful_test_marks_online(self):
        with mock.patch("cameras.admin.rtsp.test_connection", return_value=(True, "")):
            self.client.post(reverse("admin:cameras_camera_add"), {
                "name": "Front door", "rtsp_url": RTSP_URL,
            })
        camera = Camera.objects.get(name="Front door")
        self.assertEqual(camera.status, Camera.ONLINE)
        self.assertEqual(camera.last_error, "")
        self.assertIsNotNone(camera.last_checked_at)

    def test_failed_test_marks_offline_but_still_saves(self):
        with mock.patch("cameras.admin.rtsp.test_connection",
                        return_value=(False, "Could not open stream.")):
            self.client.post(reverse("admin:cameras_camera_add"), {
                "name": "Back yard", "rtsp_url": RTSP_URL,
            })
        camera = Camera.objects.get(name="Back yard")
        self.assertEqual(camera.status, Camera.OFFLINE)
        self.assertIn("Could not open stream", camera.last_error)


class AdminRenderTests(TestCase):
    def setUp(self):
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")

    def test_index_shows_cameras_section(self):
        resp = self.client.get(reverse("admin:index"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Cameras")

    def test_list_shows_masked_url(self):
        Camera.objects.create(name="Lobby", rtsp_url=RTSP_URL)
        resp = self.client.get(reverse("admin:cameras_camera_changelist"))
        self.assertContains(resp, "••••")
        self.assertNotContains(resp, "AxProVideo2024")


class LiveStreamTests(TestCase):
    def setUp(self):
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")
        self.camera = Camera.objects.create(name="Lobby", rtsp_url=RTSP_URL)

    def test_mjpeg_requires_login(self):
        self.client.logout()
        resp = self.client.get(reverse("admin:cameras_camera_mjpeg"), {"camera": self.camera.pk})
        self.assertEqual(resp.status_code, 302)

    def test_mjpeg_unknown_camera_404s(self):
        resp = self.client.get(reverse("admin:cameras_camera_mjpeg"), {"camera": 999999})
        self.assertEqual(resp.status_code, 404)

    def test_mjpeg_content_type(self):
        resp = self.client.get(reverse("admin:cameras_camera_mjpeg"), {"camera": self.camera.pk})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "multipart/x-mixed-replace; boundary=frame")

    def test_live_view_renders_camera(self):
        resp = self.client.get(reverse("admin:cameras_camera_live"), {"camera": self.camera.pk})
        self.assertContains(resp, "Lobby")
        self.assertContains(resp, f"?camera={self.camera.pk}")

    def test_live_view_unknown_camera_404s(self):
        resp = self.client.get(reverse("admin:cameras_camera_live"), {"camera": 999999})
        self.assertEqual(resp.status_code, 404)

    def test_change_page_shows_preview_and_link(self):
        resp = self.client.get(reverse("admin:cameras_camera_change", args=[self.camera.pk]))
        self.assertContains(resp, "Open full live view")
        self.assertContains(resp, f"/admin/cameras/camera/mjpeg/?camera={self.camera.pk}")

    def test_action_requires_single_selection(self):
        Camera.objects.create(name="Back yard", rtsp_url=RTSP_URL.replace("101", "102"))
        resp = self.client.post(reverse("admin:cameras_camera_changelist"), {
            "action": "view_live_stream",
            "_selected_action": [str(c.pk) for c in Camera.objects.all()],
        }, follow=True)
        self.assertContains(resp, "Select exactly one camera")

    def test_action_redirects_to_live_view(self):
        resp = self.client.post(reverse("admin:cameras_camera_changelist"), {
            "action": "view_live_stream",
            "_selected_action": [str(self.camera.pk)],
        })
        self.assertRedirects(
            resp, reverse("admin:cameras_camera_live") + f"?camera={self.camera.pk}",
        )
