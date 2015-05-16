from django.shortcuts import render
from django.http import HttpResponse

from main.forms import UploadForm

from PIL import Image

# Create your views here.
def index(request):
	return render(request, 'index.html', {'form': UploadForm()})

def upload(request):
	form = UploadForm(request.POST, request.FILES)
	im = Image.open(request.FILES['picture'])
	im.show()
	return HttpResponse(request.FILES)

