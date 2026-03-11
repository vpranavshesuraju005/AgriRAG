from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("index.html", views.index, name="index"),
    path("Signup.html", views.Signup, name="Signup"),
    path("SignupAction", views.SignupAction, name="SignupAction"),
    path("UserLogin.html", views.UserLogin, name="UserLogin"),
    path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
    path("LoadDataset.html", views.LoadDataset, name="LoadDataset"),
    path("LoadDatasetAction", views.LoadDatasetAction, name="LoadDatasetAction"),
    path("TrainML", views.TrainML, name="TrainML"),
    path("Predict", views.Predict, name="Predict"),
    path("PredictAction", views.PredictAction, name="PredictAction"),
    path("ViewSeeds", views.ViewSeeds, name="ViewSeeds"),
    path("ChatAction", views.ChatAction, name="ChatAction"),
    path("Logout", views.Logout, name="Logout"),
]