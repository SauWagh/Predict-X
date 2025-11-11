from django.urls import path
from . import views

urlpatterns = [
    path('train/', views.home, name='home'),
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/<int:pk>/', views.dataset_detail, name='dataset_detail'),
    path('predict/<int:pk>/', views.prediction, name='prediction'),
]
