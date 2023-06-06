from django.shortcuts import render
from myapp.models import Sangdata

# # Create your views here.
# def goodFunc(request):
#     return render(request, 'show.html', {'kor':85, 'eng':93})

def MainFunc(request):
    datas = Sangdata.objects.all().order_by('-code')
    return render(request, 'main.html', {'data':datas})