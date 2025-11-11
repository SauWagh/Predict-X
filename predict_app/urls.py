from django.urls import path
from predict_app.views import*

urlpatterns = [
    path('',home, name='home'),
    path('index/',upload_file, name='upload_file'),
    path('results/',results_view, name='results'),
]
