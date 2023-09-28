from django.contrib import admin
from .models import Post # 모델을 불러옴

admin.site.register(Post) # 모델 등록