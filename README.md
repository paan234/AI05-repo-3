# Identifying cracks on concrete with Image Classification

## 1. Summary of the project
The aim of this porject is to create a convolutional neural network model that can identify cracks on concrete with a high accuracy.
The problem is modelled as a binary classification problem (negative and positive crack image). 
The model is trained with a dataset of 40,000 images (20,000 images of concrete in good condition and 20,000 images og concrete with cracks).

The model is trained with dataset from [Concrete](https://data.mendeley.com/datasets/5y9wdsg2zt/2).

## 2. IDE and Framework
This project is created using Spyder as the main IDE. The main frameworks used in this project are:
- Numpy
- Matplotlib
- TensorFlow Keras

## 3. Methodology

### _3.1 Data Pipeline_
The image data are loaded along with their corresponding labels. The data is first split into train-validation set, with a ratio of 70:30.
The validation data is then further split into two portion to obtain some test data, with a ratio of 80:20. The overall train-validation-test
split ratio is 70:24:6. No data augmentation is applied as the data size and variation are already sufficient.

### _3.2 Model Pipeline_
The input layer is designed to receive coloured images with a dimension of 160x160. The full shape will be (160,160,3)

Transfer learning is applied for building the deep learning model of this project. Firstly, a preprocessing layer is created that will change the pixel values of input images to a range of -1 to 1. 
This layer serves as the feature scaler and it is also requirement for the transfer learning model to output the correct signals.

For feature extractor, a pretrained model of MobileNet v2 is used. The model is readily available within TensorFlow Keras package, with ImageNet pretrained parameters.
It is also frozen hence will not update during model training.

A global average pooling and dense layer are used as the classifier to output softmax signals. The softmax signals are used to identify the predicted class.

The simplified illustration of the model is shown in the figure below.

![alt text](https://github.com/paan234/AI05-repo-3/blob/main/Image/Model.png )

The model is trained with a batch size of 16 and 10 epochs. After training, the model reaches 99% training accuracy and 95% validation accuracy. 
The training results are shown in the figures below.

![alt text](https://github.com/paan234/AI05-repo-3/blob/main/Image/loss_graph.png)

![alt text](https://github.com/paan234/AI05-repo-3/blob/main/Image/accuracy_graph.png)

## 4. Results
The model is evaluated with the test data. The loss and accuracy are shown in figure below.

![alt text](https://github.com/paan234/AI05-repo-3/blob/main/Image/Test_result.png)

Some predictions are also been made with the model, and compared with the actual results.

![alt text](https://github.com/paan234/AI05-repo-3/blob/main/Image/Result.png)
