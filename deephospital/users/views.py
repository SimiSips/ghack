from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, F"Your account has been created for {username}! You are now able to login!")
            return redirect('admin')
    else:
        form = UserRegisterForm()
    return render(request, 'users/register.html', {'form' : form})