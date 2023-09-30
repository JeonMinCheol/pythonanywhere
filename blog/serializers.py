from blog.models import Post
from rest_framework import serializers
class PostSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = ['id','author','title', 'text', 'created_date', 'published_date', 'image']