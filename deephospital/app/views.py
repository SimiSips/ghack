# Import necessary libaries
from tensorflow.keras.models import load_model
import cv2 
import numpy as np 
from django.shortcuts import render

# Load the machine learning model
model = load_model("deephospital_model")

def prediction(request):