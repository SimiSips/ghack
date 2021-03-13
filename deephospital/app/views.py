from django.shortcuts import render
from .forms import ImageForm
from tensorflow.keras.models import load_model


def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            return render(request, 'main/upload.html', {'form': form, 'img_obj': img_obj})
            image = form.cleaned_data['image']
    else:
        form = ImageForm()
    return render(request, 'main/upload.html', {'form': form})