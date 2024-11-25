from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('signup/', views.signup, name='signup'),
    path('upload/', views.upload, name='upload'),
    path('results/<int:pk>/', views.results, name='results'),
    path('login/', views.CustomLoginView.as_view(), name='login'),
    path('logout/', views.CustomLogoutView.as_view(), name='logout'),
]
