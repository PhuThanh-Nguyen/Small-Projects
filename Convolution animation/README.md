<i> Note: </i> Để tạo ảnh .gif từ các ảnh tạo ra từ convolution_animation.py thực hiện các bước sau: (yêu cầu đã cài sẵn ffmpeg)
1. Chuyển ảnh sang video
```
ffmpeg -f image2 -i t_%d.jpg video.avi
```
2. Chuyển video sang ảnh .gif
```
ffmpeg -i video.avi -pix_fmt gray animation.gif
```
Tham khảo từ: <a> https://stackoverflow.com/questions/3688870/create-animated-gif-from-a-set-of-jpeg-images </a>
