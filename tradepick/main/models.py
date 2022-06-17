from django.db import models
from sklearn.metrics import max_error

# Create your models here.
class ticker(models.Model):
    fullname = models.CharField(max_length=164)
    tickername = models.CharField(max_length=50)
    def __str__(self):
        return self.fullname