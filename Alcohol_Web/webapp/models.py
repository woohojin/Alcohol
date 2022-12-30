from django.db import models


class Alcohol(models.Model):
    name = models.CharField(max_length=50)
    abv = models.FloatField()
    price = models.IntegerField()
    standard = models.CharField(max_length=50)
    material = models.CharField(max_length=256)
    company = models.CharField(max_length=50)
