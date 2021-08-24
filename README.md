# glucoma-image-classification-ConvNet

The high-level way a convolution neural network works is as follows-


![alt text](https://github.com/niteshjindal170988/glucome-image-classification-ConvNet/blob/main/cnn_flow.JPG?raw=true)


Referring to our example and the given illustration, we have input retinal images of glucoma(infected which we name as positive images) and there are normal retinal images(non-infected which we name as negative images). We are applying Convolution neural network to classify or distinguish between normal and glucoma retinal eye image on unknown or test images.

Dataset which is being used in the given repository can be found at below link-  

https://ieee-dataport.org/documents/1450-fundus-images-899-glaucoma-data-and-551-normal-data

However, for demonstration purpose, we have taken a sample images from the given path that can be accessed through below link-

https://drive.google.com/drive/folders/1PiD-fEoBVfvNUmhsmk4lVMxwbqKIiZXd?usp=sharing 

, collating all the postive and negative images.
        That is, temp_imgs contains all positive+negative images, whereas temp_labels contains all postive+negative image labels.
        Please note that images are labelled 1 for positive and 0 for negative.


In our example, we defined all pathnames where we have stored train data and test data.Then, we collated all the postive and negative images in a directory.We manage to label the images with 1 being positive image and 0 being negative/normal image.

For reading images, we use `imread()` function from opencv module and stored then as 4D tensor of shape ( samples, height, width, channels). Since, these are colored images, we have three channels(R,G,B). We then resized the image to shape (150,150,3) and there were such 156 sample as evident in [glucoma-image-classification.ipynb].

In the `model compilation stage`,we created a sequential model which is considered appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. We added convolution 2D filters, max pooling layers followed by flattening the convolution output the input before passing it as an input to full connected dense layer.

To explain it in detail, the convolution 2D filters learn separate features in the image therefore we have multiple conv2D filters within a single layer.

If we glance at the code[glucoma-image-classification.ipynb], we notice that in first layer, we have 16 conv2D filters followed by 32 kernels in the second layer followed by 64 kernels in the third layer or final layer.

**We are passing below arguments to `Conv2D` function in the first layer are -**

filters/kernels = 16 

kernel_size= 3*3 determines the width and height of the convolution filter window. Usually we keep kernels as powers of 2 and though our dataset is not so huge, we have used `Conv2D` filters in the range [16,32,64].

activation = relu function which stands for Rectified Linear Unit i.e. `f(x)=max(0,x)`.This function has a property that for the negative pixels, output is zero. Therefore, it activates certain number of neurons.The purpose of applying the rectifier function is to increase the non-linearity in our images.

input_shape= input image shape, which is (150,150,3).

Then, we have MaxPooling that reduces the spatial size of convolved features.That is, it reduces the dimensions of a image by extracting the maximum value from the image portion captured by a max pool 2*2 filter. Then, flattening the output of convolution network layers to convert the data in 1-dimensional form before connecting it to the fully- connected layer followed by the output layer. Lastly, activation function in the output is sigmoid function because of the binary response.

Before training part, we have used `ImageDataGenerator` which accepts the input data, randomly transforms it, and returns the new transformed data. To put it another way, we do not get the original data but the transformed data i.e. input data is slightly modified versions of the original input data and the network is able to learn more robust features.


While training, we have used the `ModelCheckpoint()` which saves the best model of the several iterations where the loss is minimum.Model will be saved to the path if there happens any improvement in loss.
As per our results, we observe that the loss has consistently decreased and was 0.211369 in final epoch which is actually good. However, results on the test set were not so good which implies that our model is overfitted.

Accuracy in final epoch on train set=0.96; Accuracy on test set=0.75.

This difference can be attributed to the imbalanced dataset as well as the size of dataset.However, it can be further improved in many ways. One way is to develop algorithm to segment postive and negative glucoma images from the repository. Once we get the huge labelled data and train model on it, the difference will certainly shrink. Also, tuning of the filter parameters definitely improves the model performance on test set.


**References:-**

https://arxiv.org/ftp/arxiv/papers/1506/1506.01195.pdf

https://keras.io/api/layers/convolution_layers/convolution2d/

https://arxiv.org/abs/2004.13529  

https://ieee-dataport.org/documents/1450-fundus-images-899-glaucoma-data-and-551-normal-data

https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0263-7

https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/
