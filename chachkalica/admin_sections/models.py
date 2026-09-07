"""Cross-cutting "admin section" tagging.

Lets any admin-registered model's rows be partitioned across multiple blank
admin fronts (see admin_sections/sites.py) without a migration on that model's
own table.

An ``AdminSection`` is one project's own admin front, reachable at
``/admin/s/<slug>/`` — mounted by the single dynamic ``section_site`` in
sites.py, so adding one is a DB row, not a urls.py edit or a redeploy. A
``SectionAssignment`` is the one row that says "this object currently lives in
that section", keyed by ContentType so it works for any model uniformly. No
assignment row means the object still shows up on the main ``/admin/``
(unfiltered, exactly as before this feature existed) but on no section front.
"""

from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models


class AdminSection(models.Model):
    slug = models.SlugField(unique=True, help_text="Used in the URL: /admin/s/<slug>/")
    name = models.CharField(max_length=100, help_text="Shown in the top nav and as this admin's header.")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class SectionAssignment(models.Model):
    """Which section (if any) one specific row currently belongs to.

    ``object_id`` is a plain CharField rather than tied to the target model's
    own pk type/field, so this works the same whether the target uses an
    integer or a UUID primary key.
    """

    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.CharField(max_length=64)
    content_object = GenericForeignKey("content_type", "object_id")
    section = models.ForeignKey(AdminSection, on_delete=models.CASCADE, related_name="assignments")

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["content_type", "object_id"], name="one_section_per_object"
            ),
        ]

    def __str__(self):
        return f"{self.content_type} #{self.object_id} -> {self.section}"
