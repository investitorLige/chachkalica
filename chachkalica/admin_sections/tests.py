from django.contrib.auth import get_user_model
from django.test import TestCase

from admin_sections.models import AdminSection, SectionAssignment
from fleet.models import Annotator

User = get_user_model()


class SectionScopingTests(TestCase):
    """Exercises the mixin apps.py injects into every registered ModelAdmin:
    unassigned rows only ever show on the main /admin/, and an AdminSection's
    own front only shows what's been transferred into it."""

    def setUp(self):
        User.objects.create_superuser("admin", "admin@example.com", "pw")
        self.client.login(username="admin", password="pw")
        self.section = AdminSection.objects.create(slug="proj-x", name="Project X")
        self.a1 = Annotator.objects.create(username="a1")
        self.a2 = Annotator.objects.create(username="a2")

    def test_new_section_front_starts_blank(self):
        main = self.client.get("/admin/fleet/annotator/")
        section = self.client.get("/admin/s/proj-x/fleet/annotator/")

        self.assertEqual(main.status_code, 200)
        self.assertEqual(section.status_code, 200)
        self.assertEqual(main.context["cl"].result_count, 2)
        self.assertEqual(section.context["cl"].result_count, 0)

    def test_transfer_moves_row_into_the_section(self):
        resp = self.client.post(
            "/admin/fleet/annotator/",
            {
                "action": "transfer_to_section",
                "_selected_action": [str(self.a1.pk)],
                "apply": "1",
                "section": str(self.section.pk),
            },
        )
        self.assertEqual(resp.status_code, 302)  # redirected back to the changelist
        self.assertTrue(
            SectionAssignment.objects.filter(object_id=str(self.a1.pk), section=self.section).exists()
        )

        section_page = self.client.get("/admin/s/proj-x/fleet/annotator/")
        self.assertEqual(section_page.context["cl"].result_count, 1)
        self.assertEqual(section_page.context["cl"].result_list[0], self.a1)

        # Unfiltered main admin still shows everything — a transfer tags, it
        # doesn't remove the row from anywhere else.
        main_page = self.client.get("/admin/fleet/annotator/")
        self.assertEqual(main_page.context["cl"].result_count, 2)

    def test_transfer_with_no_destination_clears_the_assignment(self):
        SectionAssignment.objects.create(
            content_type=self._annotator_ct(), object_id=str(self.a1.pk), section=self.section
        )

        self.client.post(
            "/admin/fleet/annotator/",
            {
                "action": "transfer_to_section",
                "_selected_action": [str(self.a1.pk)],
                "apply": "1",
                "section": "",
            },
        )

        self.assertFalse(SectionAssignment.objects.filter(object_id=str(self.a1.pk)).exists())

    def test_intermediate_page_renders_without_apply(self):
        resp = self.client.post(
            "/admin/fleet/annotator/",
            {"action": "transfer_to_section", "_selected_action": [str(self.a1.pk)]},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Transfer to admin section")
        self.assertContains(resp, self.section.name)

    def test_anonymous_is_redirected_to_login_on_a_section_front(self):
        # Redirects to the *section's own* login, not the main admin's — each
        # front is a separate AdminSite instance with its own login view.
        self.client.logout()
        resp = self.client.get("/admin/s/proj-x/fleet/annotator/")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/admin/s/proj-x/login/", resp.url)

    def test_creating_a_section_redirects_straight_to_its_new_front(self):
        resp = self.client.post(
            "/admin/admin_sections/adminsection/add/",
            {"slug": "proj-y", "name": "Project Y"},
        )
        self.assertRedirects(resp, "/admin/s/proj-y/", fetch_redirect_response=False)

    @staticmethod
    def _annotator_ct():
        from django.contrib.contenttypes.models import ContentType

        return ContentType.objects.get_for_model(Annotator)
