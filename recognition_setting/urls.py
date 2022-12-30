from django.urls import re_path
from django.contrib import admin

from recognition_api import mnist

urlpatterns = [
	re_path(r'^admin/', admin.site.urls),
	re_path(r'^$', mnist.page, name="index"),
]
