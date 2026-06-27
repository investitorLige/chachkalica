"""Webhook endpoints contributed by the fleet app."""

from django.urls import path

from fleet import views

urlpatterns = [
    path("hook", views.hook, name="hook"),
    path("health", views.health, name="health"),
]
