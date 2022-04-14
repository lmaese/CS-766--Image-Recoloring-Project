from django.http import HttpResponse
from django.shortcuts import render
from django import forms
from .models import Image
from .functions import transform
import PIL
from numpy import asarray

class ImageForm(forms.ModelForm):
    """Form for the image model"""
    class Meta:
        model = Image
        fields = ('title', 'image')

def index(request):
    #return render(request, 'color/index.html', {})
    if request.method == 'POST':
        
        print(request.POST)
        print(request.FILES)
        print(asarray(PIL.Image.open(request.FILES['image'])).shape)
        img_gray = transform.get_gray(asarray(PIL.Image.open(request.FILES['image'])))
        print({'image_gray':img_gray})
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            img = img_obj.image
            img_gray = img_obj.image_gray

            print(form.instance.image)

            img_loc_gray = transform.get_gray(img_loc)
            img_gray = ImageForm(request.POST, {'image_gray':('1.jpg', open("media/" + img_loc_gray, "rb"))})
            if img_gray.is_valid():
                img_gray.save()
            else:
                print('not valid!')

            
            return render(request, 'color/index.html', {'form': form, 'img': img_loc, 'img_gray': img_gray.instance.image})
        else:
            print('not valid!')
    else:
        form = ImageForm()
    return render(request, 'color/index.html', {'form': form})



