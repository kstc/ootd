import numpy as np
import hashlib

from django.shortcuts import render
from django.templatetags.static import static
from django.conf import settings
from django.views.generic.edit import FormView
from django.core.urlresolvers import reverse_lazy

from main.forms import UploadForm
from main.feature_extraction import get_dominant_color
from main.kmeans import load_kmeans, get_similar

from PIL import Image

# Create your views here.
def index(request):
    return render(request, 'index.html', {'form': UploadForm()})

def upload(request):
    try:
        form = UploadForm(request.POST, request.FILES)
        if not form.is_valid():
            return render(request, 'index.html', {'form': UploadForm()})

        # Get form details
        sex = request.POST.get('sex')
        clothing_type = request.POST.get('men_clothing_type') if sex == '1'\
                        else request.POST.get('women_clothing_type')
        image = request.FILES['picture']

        im = Image.open(image)

        # Save uploaded image
        uploaded_filename = str(image)
        h = hashlib.new('ripemd160')
        h.update(uploaded_filename)
        uploaded_filename = h.hexdigest() + '.png'
        # Be sure to configure absolute path to 'uploaded' folder in settings.py
        im.save(settings.UPLOAD_DIRECTORY + uploaded_filename)

        input_data = [get_dominant_color(im), clothing_type]

        if sex == '1':
            kmeans = load_kmeans('mens10.km')
            dataset = 'data_set_men.csv'
        elif sex == '2':
            kmeans = load_kmeans('womens4.km')
            dataset = 'data_set_women.csv'

        similar_items = get_similar(input_data, kmeans, dataset)
        similar_items = [static('img/' + i) for i in similar_items]
        return render(request, 'index.html', {'form': UploadForm(request.POST, request.FILES),
                                             'img': similar_items,
                                             'uploaded': static('uploaded/' + uploaded_filename)})
    except:
        return render(request, 'index.html', {'form': UploadForm()})
