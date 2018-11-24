"""mysite URL Configuration
"""
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('posts/', include('posts.urls')),
    path('polls/', include('polls.urls')),
    path('admin/', admin.site.urls),
]
