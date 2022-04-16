from django.http import HttpResponse
from django.shortcuts import render
from django import forms
from .models import Image
from .functions import transform
import PIL
from numpy import asarray
from django.core.files.base import ContentFile
from io import StringIO
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import BytesIO

class ImageForm(forms.ModelForm):
    """Form for the image model"""
    class Meta:
        model = Image
        fields = ('image', )

def index(request):
    #return render(request, 'color/index.html', {})
    if request.method == 'POST':
        
        print(request.POST)
        print(request.FILES)
        
        #
        
        form = ImageForm(request.POST, request.FILES)
        print(request.POST)
        print(request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            img = img_obj.image
            #img_gray = img_obj.image_gray

            print(img_obj.title)
            print(img_obj.image)
            print(img_obj.image_gray)

            thumb = transform.get_gray(asarray(PIL.Image.open(img_obj.image)))
            buffer = BytesIO()
            thumb.save(fp=buffer, format='JPEG')
            buff_val = buffer.getvalue()
            img_obj.image_gray.save('foo.jpg', ContentFile(buff_val))
            
            # thumb_io = StringIO()
            # thumb.save(thumb_io, format='JPEG')
            # thumb_file = InMemoryUploadedFile(thumb_io, None, 'foo.jpg', 'image/jpeg',
            #                       thumb_io.len, None)

            # img_obj.image_gray.save('foo.jpg', ContentFile(thumb_file.read()))
            img_obj.save()
            print(img_obj.image_gray)

            #img_loc_gray = transform.get_gray(img_loc)
            #img_gray = ImageForm(request.POST, {'image_gray':('1.jpg', open("media/" + img_loc_gray, "rb"))})

            
            return render(request, 'color/index.html', {'form': form, 'img': img_obj.image, 'img_gray': img_obj.image_gray})
        else:
            print('not valid!')
    else:
        form = ImageForm()
    return render(request, 'color/index.html', {'form': form})



