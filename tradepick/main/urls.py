from django import urls
from django.urls import URLPattern, path

from . import views

urlpatterns = [
    path("",views.index, name="index"),
    path("processstock", views.processstock, name="processstock"),
]