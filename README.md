# glucome-image-classification-ConvNet
The high-level way a convolution neural network works is as follows-


![alt text](https://github.com/niteshjindal170988/glucome-image-classification-ConvNet/blob/main/cnn_flow.JPG?raw=true)


Referring to our example and the given illustration, we have input retinal images of glucoma(infected which we name as positive images) and there are normal retinal images(non-infected which we name as negative images). We are applying Convolution neural network to classify or distinguish between normal and glucoma retinal eye image on unknown or test images.

Dataset which is being used in the given repository can be found at below link-  

https://ieee-dataport.org/documents/1450-fundus-images-899-glaucoma-data-and-551-normal-data

However, for demonstration purpose, we have taken a sample images from the given path that can be accessed through below link-

https://drive.google.com/drive/folders/1PiD-fEoBVfvNUmhsmk4lVMxwbqKIiZXd?usp=sharing 

In our example, we are developing a sequential model which is considered appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. We added convolution 2D filters, max pooling layers followed by flattening the convolution output the input before passing it as an input to full connected dense layer.
To explain it in detail, the convolution 2D filters learn separate features in the image therefore we have multiple conv2D filters within a single layer.
If we glance at the code[glucoma-image-classification.ipynb], we notice that in first layer, we have 16 conv2D filters followed by 32 kernels in the second layer followed by 64 kernels in the third layer or final layer.
**We are passing below arguments to `Conv2D` function in the first layer are -**
**filters/kernels"" = 16 
kernel_size= 3*3 determines the width and height of the convolution filter window
activation = relu function
input_shape= input image shape which is (150,150,3).
