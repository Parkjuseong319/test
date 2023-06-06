from django.db import models

# Create your models here.
class SurveyData(models.Model):
    job = models.CharField(max_length=50)
    gender = models.CharField(max_length=10)
    game_time = models.FloatField(default=0)