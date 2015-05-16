import numpy as np

from django.shortcuts import render
from django.http import HttpResponse
from django.templatetags.static import static

from main.forms import UploadForm
from main.feature_extraction import get_dominant_color
from main.kmeans import load_kmeans, get_similar

from PIL import Image

# Create your views here.
def index(request):
	return render(request, 'index.html', {'form': UploadForm()})

def upload(request):
	form = UploadForm(request.POST, request.FILES)
	im = Image.open(request.FILES['picture'])
	sex = request.POST.get('sex')
	input_data = []
	clothing_type = request.POST.get('clothing_type')
	dom_color = get_dominant_color(im)
	input_data.append([dom_color, clothing_type])
	kmeans = load_kmeans('mens.km') if sex == '1' else load_kmeans('womens.km')
	dataset = 'data_set_men.csv' if sex == '1' else 'data_set_women.csv'
	same = get_similar(input_data, kmeans, dataset)
	same = [static(i) for i in same]
	print same
	return render(request, 'index.html', {'form': UploadForm(), 'img': same})


