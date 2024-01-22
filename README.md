# AI_COVID-19

## Dataset
Our dataset has 2314 images consisting of 1220 images of people wearing a mask and 1094 images without a mask. 

Due to the non-availability of the dataset containing masks, we have created the dataset by taking an ordinary image of a face and an artificial mask image. 

This artificial mask is applied to the face image using computer vision techniques thereby, creating the category of images of people wearing a mask.

## How did we create the dataset?

To create the Dataset :
  1. Take normal images of faces.
  2. Create a custom computer vision Python script to add face masks to them, thereby creating an artificial (but still real-world applicable) dataset.

Without Mask

![example_02](https://github.com/Samhita-kolluri/AI_COVID-19/assets/65637090/fdfb56d6-5319-49bb-b709-6d9718f88302)

With Mask

![example_01](https://github.com/Samhita-kolluri/AI_COVID-19/assets/65637090/7f947fc9-3780-4ed0-b589-078523dc6df5)

## Steps to create the dataset
  1. Start with an image of a person NOT wearing a face mask.
  2. Apply face detection using OpenCV to compute the location of the bounding box in the frame.
  3. After locating the face in the frame we extract the face ROI using OpenCV and NumPy.
  4. Later on, we apply facial landmarks to localize the facial structures.
  5. A transparent background image of a mask is considered that will be automatically applied to the face by using the facial landmarks.
  6. The mask is then resized and rotated by placing it on the face.
  7. This process is repeated for all the input images, creating our artificial face
dataset.

## Image Preprocessing:

Initially, we used ImageDataGenerator to generate images in various orientations to create a batch of images. These generated batches contain the normalized data, which speeds up the learning process of the model. 

The LabelBinarizer preprocessing technique is used to gather the essential features from the image and generalize the model accurately. 

Our system first collects a frame from the video stream and is fed into two models after reshaping the image size to 224x224 with three channels (RBG).

In the next step, we convert these image pixels to an array by ensuring the intensity of each pixel in the input falls between the range [-1, 1], and then these preprocessed images are appended to the data with their corresponding binary labels in the form of lists. 

The labels are encoded by using the One-Hot Encoding method to achieve the best classification performance.


## Architecture Description

We have used two different models. 

The first model is responsible for locating the faces in a frame and subsequently checking for social distancing among the located faces of individuals. The second model is to classify the image whether the specific individual is wearing a mask. In the first model, the face location is detected using various OpenCV techniques. The new version of OpenCV contains a Deep Neural Network (DNN) module that can support different frameworks such as Tensor Flow, Caffe, and PyTorch with pre-trained weights, making it easier to load models from the disk. The DNN module takes the loaded model as input and selects the backend process. The loaded model is built using the **SingleShot-Multibox detector (SSD) for face detection** and **ResNet-10 architecture** as the backbone. After loading the model with the pre-trained weights, the DNN model detects the face from the frame with a resized width shape of 400.

The second model is used for the **binary classification** of images into individuals with and without masks. We have used **MobileNetV2 architecture** as our base model with pre-trained ImageNet weights. MobileNet uses depth-wise separable convolutions and is built to take lightweight DNNs by introducing two global hyperparameters. MobileNetV1 uses depthwise separable convolutions and point convolutions by scaling the width and resolution of the image. By careful tuning of these hyperparameters, the complexity and the model parameters are reduced.
On the other hand, in MobileNetv2, the architecture is represented in three structures and the depthwise separable convolution is divided into two parts. 
1. depthwise separable convolutions
2. linear bottleneck
3. Inverted residual block.


## Results

The proposed system combines AI and IoT technologies and showcases their potential in combating covid-19. For our training, the model is initialized with proper initializes. The proposed classification model is trained with a batch size of 32 and a learning rate of 1e-4 for 20 epochs. Our model got an accuracy of 99% with a minimal loss of 0.071. The precision for with and without mask classes are 0.999 and 1.00, respectively, and an F1 score of 0.99 was obtained for both classes. The validation accuracy and loss of the model are obtained

![plot](https://github.com/Samhita-kolluri/AI_COVID-19/assets/65637090/f66fe72b-19f0-43b7-83b8-be8a7e9fff5c)

