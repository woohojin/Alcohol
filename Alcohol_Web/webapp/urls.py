# -*- coding: utf-8 -*-


from django.urls import path
from . import views

urlpatterns = [
    path("image/", views.image, name="image"),
    path("camera/", views.camera, name="camera"),
    path("predict/", views.predict, name="predict"),
    path("predict_camera/", views.predict_camera, name="predict_camera"),
    path("picture/", views.picture, name="picture"),
]
