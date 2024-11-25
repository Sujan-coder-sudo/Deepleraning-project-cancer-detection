from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .models import MedicalImage
from .forms import MedicalImageForm
from .ml_model import predict_cancer
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import reverse_lazy

# Include the LoginView in your app
class CustomLoginView(LoginView):
    template_name = 'cancer_detection/login.html'
    redirect_authenticated_user = True

# Logout view
class CustomLogoutView(LogoutView):
    next_page = reverse_lazy('home')

def home(request):
    return render(request, 'cancer_detection/home.html')

def about(request):
    return render(request, 'cancer_detection/about.html')

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'cancer_detection/signup.html', {'form': form})

@login_required
def upload(request):
    if request.method == 'POST':
        form = MedicalImageForm(request.POST, request.FILES)
        if form.is_valid():
            medical_image = form.save(commit=False)
            medical_image.user = request.user
            medical_image.save()
            
            # Perform prediction
            prediction = predict_cancer(medical_image.image.path)
            medical_image.prediction = prediction
            medical_image.save()
            
            return redirect('results', pk=medical_image.pk)
    else:
        form = MedicalImageForm()
    return render(request, 'cancer_detection/upload.html', {'form': form})

@login_required
def results(request, pk):
    medical_image = MedicalImage.objects.get(pk=pk)
    return render(request, 'cancer_detection/result.html', {'medical_image': medical_image})



# Create your views here.
