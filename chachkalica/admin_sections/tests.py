import tempfile
from pathlib import Path

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from admin_sections.models import AdminSection, SectionAssignment
from fleet.models import Annotator, Dataset, FleetSettings

User = get_user_model()


class SectionScopingTests(TestCase):
    """Exercises the mixin apps.py injects into every registered ModelAdmin:
    unassigned rows only ever show on the main /admin/, an AdminSection's own
    front only shows what's been added to it, and a row can belong to several
    sections (plus main) at once — this is membership, not a move."""

    def setUp(self):
        User.objects.create_superuser("admin", "admin@example.com", "pw")
        self.client.login(username="admin", password="pw")
        self.section = AdminSection.objects.create(slug="proj-x", name="Project X")
        self.other_section = AdminSection.objects.create(slug="proj-y", name="Project Y")
        self.a1 = Annotator.objects.create(username="a1")
        self.a2 = Annotator.objects.create(username="a2")

    def test_new_section_front_starts_blank(self):
        main = self.client.get("/admin/fleet/annotator/")
        section = self.client.get("/admin/s/proj-x/fleet/annotator/")

        self.assertEqual(main.status_code, 200)
        self.assertEqual(section.status_code, 200)
        self.assertEqual(main.context["cl"].result_count, 2)
        self.assertEqual(section.context["cl"].result_count, 0)

    def _add(self, pk, section, from_path="/admin/fleet/annotator/"):
        return self.client.post(
            from_path,
            {
                "action": "add_to_section",
                "_selected_action": [str(pk)],
                "apply": "1",
                "section": str(section.pk),
            },
        )

    def test_add_puts_the_row_on_the_section_front_without_removing_it_from_main(self):
        resp = self._add(self.a1.pk, self.section)
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(
            SectionAssignment.objects.filter(object_id=str(self.a1.pk), section=self.section).exists()
        )

        section_page = self.client.get("/admin/s/proj-x/fleet/annotator/")
        self.assertEqual(section_page.context["cl"].result_count, 1)
        self.assertEqual(section_page.context["cl"].result_list[0], self.a1)

        main_page = self.client.get("/admin/fleet/annotator/")
        self.assertEqual(main_page.context["cl"].result_count, 2)

    def test_a_row_can_be_added_to_a_second_section_while_staying_in_the_first(self):
        self._add(self.a1.pk, self.section)
        self._add(self.a1.pk, self.other_section, from_path="/admin/s/proj-x/fleet/annotator/")

        self.assertEqual(
            set(SectionAssignment.objects.filter(object_id=str(self.a1.pk)).values_list("section__slug", flat=True)),
            {"proj-x", "proj-y"},
        )
        self.assertEqual(self.client.get("/admin/s/proj-x/fleet/annotator/").context["cl"].result_count, 1)
        self.assertEqual(self.client.get("/admin/s/proj-y/fleet/annotator/").context["cl"].result_count, 1)
        self.assertEqual(self.client.get("/admin/fleet/annotator/").context["cl"].result_count, 2)

    def test_adding_the_same_row_to_the_same_section_twice_is_a_no_op(self):
        self._add(self.a1.pk, self.section)
        self._add(self.a1.pk, self.section)  # should not raise (unique constraint) or duplicate
        self.assertEqual(
            SectionAssignment.objects.filter(object_id=str(self.a1.pk), section=self.section).count(), 1
        )

    def test_remove_from_this_section_only_detaches_the_current_one(self):
        self._add(self.a1.pk, self.section)
        self._add(self.a1.pk, self.other_section, from_path="/admin/s/proj-x/fleet/annotator/")

        self.client.post(
            "/admin/s/proj-x/fleet/annotator/",
            {"action": "remove_from_this_section", "_selected_action": [str(self.a1.pk)]},
        )

        self.assertFalse(
            SectionAssignment.objects.filter(object_id=str(self.a1.pk), section=self.section).exists()
        )
        self.assertTrue(
            SectionAssignment.objects.filter(object_id=str(self.a1.pk), section=self.other_section).exists()
        )
        self.assertEqual(self.client.get("/admin/fleet/annotator/").context["cl"].result_count, 2)

    def test_remove_from_this_section_action_is_not_offered_on_main_admin(self):
        resp = self.client.get("/admin/fleet/annotator/")
        cl_admin = resp.context["cl"].model_admin
        self.assertNotIn("remove_from_this_section", cl_admin.get_actions(resp.wsgi_request))
        self.assertIn("add_to_section", cl_admin.get_actions(resp.wsgi_request))

    def test_intermediate_page_renders_without_apply(self):
        resp = self.client.post(
            "/admin/fleet/annotator/",
            {"action": "add_to_section", "_selected_action": [str(self.a1.pk)]},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Add to admin section")
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
            {"slug": "proj-z", "name": "Project Z"},
        )
        self.assertRedirects(resp, "/admin/s/proj-z/", fetch_redirect_response=False)

    def test_django_rq_dashboard_renders_under_a_section(self):
        # django-rq's Dashboard proxy model gets mirrored onto every section
        # like any other registered model, but its own dashboard view builds
        # its page context via admin.site.each_context(request) — hardcoded to
        # the *main* site, regardless of which site actually served the page.
        # That enumerates every app in the main site's registry (admin_sections
        # included) and needs each one to also resolve under the section's own
        # urlconf, or it 500s reversing the app-list link for it.
        resp = self.client.get("/admin/s/proj-x/django_rq/dashboard/")
        self.assertEqual(resp.status_code, 200)

    def test_visiting_a_section_does_not_leak_its_script_prefix_afterwards(self):
        # set_script_prefix is a thread-local, not per-request state — it must
        # be reset on every request (including non-section ones), not just set
        # when a section matches, or it sticks around for whatever's next.
        self.client.get("/admin/s/proj-x/fleet/annotator/")
        self.assertEqual(reverse("admin:index"), "/admin/")

    def test_adding_a_row_from_within_a_section_auto_tags_it_there(self):
        # Otherwise it would vanish from that same changelist right after
        # you created it — get_queryset only shows already-tagged rows.
        resp = self.client.post(
            "/admin/s/proj-x/fleet/annotator/add/",
            {"username": "new1", "status": "active", "_save": "Save"},
        )
        self.assertEqual(resp.status_code, 302)
        new_annotator = Annotator.objects.get(username="new1")
        self.assertTrue(
            SectionAssignment.objects.filter(
                object_id=str(new_annotator.pk), section=self.section
            ).exists()
        )
        self.assertEqual(
            self.client.get("/admin/s/proj-x/fleet/annotator/").context["cl"].result_count, 1
        )


class DuplicateDatasetAcrossSectionsTests(TestCase):
    """A Dataset's ``name`` isn't unique at the DB level any more — two rows
    may point at the same on-disk directory, as long as they never end up
    visible on the same admin front together (main admin included)."""

    def setUp(self):
        User.objects.create_superuser("admin", "admin@example.com", "pw")
        self.client.login(username="admin", password="pw")
        self.proj_x = AdminSection.objects.create(slug="proj-x", name="Project X")
        self.proj_y = AdminSection.objects.create(slug="proj-y", name="Project Y")

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        src = Path(self.tmp.name)
        (src / "shared_ds").mkdir()

        fs = FleetSettings.load()
        fs.source_dir = str(src)
        fs.save()

    def _add(self, name, path="/admin/fleet/dataset/add/"):
        return self.client.post(
            path, {"name": name, "storage_type": Dataset.LOCAL, "storage_root": "", "_save": "Save"}
        )

    def test_adding_from_a_section_scopes_the_taken_check_to_that_section(self):
        resp = self._add("shared_ds", "/admin/s/proj-x/fleet/dataset/add/")
        self.assertEqual(resp.status_code, 302)
        ds = Dataset.objects.get(name="shared_ds")
        self.assertTrue(
            SectionAssignment.objects.filter(object_id=str(ds.pk), section=self.proj_x).exists()
        )

    def test_the_same_directory_gets_an_independent_row_per_section(self):
        self._add("shared_ds", "/admin/s/proj-x/fleet/dataset/add/")
        resp = self._add("shared_ds", "/admin/s/proj-y/fleet/dataset/add/")
        self.assertEqual(resp.status_code, 302)

        self.assertEqual(Dataset.objects.filter(name="shared_ds").count(), 2)
        self.assertEqual(
            self.client.get("/admin/s/proj-x/fleet/dataset/").context["cl"].result_count, 1
        )
        self.assertEqual(
            self.client.get("/admin/s/proj-y/fleet/dataset/").context["cl"].result_count, 1
        )
        # Main is the unfiltered overview — it correctly shows both.
        self.assertEqual(
            self.client.get("/admin/fleet/dataset/").context["cl"].result_count, 2
        )

    def test_cannot_add_the_same_directory_twice_within_the_same_section(self):
        self._add("shared_ds", "/admin/s/proj-x/fleet/dataset/add/")
        resp = self._add("shared_ds", "/admin/s/proj-x/fleet/dataset/add/")
        self.assertEqual(resp.status_code, 200)  # form re-rendered with an error, not saved
        self.assertEqual(Dataset.objects.filter(name="shared_ds").count(), 1)

    def test_adding_from_main_still_excludes_names_used_anywhere(self):
        self._add("shared_ds", "/admin/s/proj-x/fleet/dataset/add/")
        resp = self._add("shared_ds", "/admin/fleet/dataset/add/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Dataset.objects.filter(name="shared_ds").count(), 1)
