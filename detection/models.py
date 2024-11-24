from django.db import models
from django.contrib.auth.models import User

class MedicalImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='medical_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"Image uploaded by {self.user.username} at {self.uploaded_at}"


# Create your models here.
