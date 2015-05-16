from django.conf.urls import patterns, include, url

from main import views

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'OOTD.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    url(r'^$', views.index, name='index'),
    url(r'^upload', views.upload, name='upload'),
)
