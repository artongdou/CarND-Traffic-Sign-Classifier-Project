# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/1_data_visualization.jpg "1_data_visualization"
[image2]: ./writeup_imgs/2_augment_img.jpg "2_augment_img"
[image3]: ./writeup_imgs/2_random_rotate.jpg "2_random_rotate"
[image4]: ./writeup_imgs/2_random_shift.jpg "2_random_shift"
[image5]: ./writeup_imgs/2_random_warp.jpg "2_random_warp"
[image6]: ./writeup_imgs/2_hist_original_train_set.jpg "2_hist_original_train_set"
[image7]: ./writeup_imgs/2_hist_augment_train_set.jpg "2_hist_augment_train_set"
[image8]: ./writeup_imgs/2_normalized_train_set.jpg "2_normalized_train_set"
[image9]: ./writeup_imgs/3_internet_imgs_prediction.jpg "3_internet_imgs_prediction"
[image10]: ./writeup_imgs/3_top5_prediction.jpg "3_top5_prediction"

<!-- [image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5" --> -->

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/artongdou/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the standard library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32,32,3)`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It randomly picks and shows one image of each available classes in the training data set. The actual class assigned to the image is shown in the title along with its index inside the data set for reference.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

After visualizing how the training set (`X_train`) is distributed, it's clear that the data set is highly unbalanced, meaning that some classes have significantly more number of sample images than the other. In order for training the network more efficiently and produce a more accurate model, I generate augmented images for each class to artificially balance the data set to have equal number of sample images of each class. `augment_img` will take an image as input and randomly `warp`, `rotate` and then `shift` the image to create an augmented image that will be added to the training set.

``` python
def augment_img(img):
    img_out = random_warp(img)
    img_out = random_rotate(img_out)
    img_out = random_shift(img_out)
    return img_out
```

Here shows the comparison between original and augmented image for `augment_img`

![alt text][image2]

Here are one example for `random_rotate`

![alt text][image3]

Here are one example for `random_shift`

![alt text][image4]

Here are one example for `random_warp`

![alt text][image5]

Here is the histogram of the original training set (`X_train`)

![alt text][image6]

Here is the histogram of the augmented training set (`X_train_new`)

![alt text][image7]

Next, `normalize_set` is implemented to normalize the training, validation, and testing set. Each image in the data set is processed the same way as the following:
1. Convert to grayscale
2. Equalize histogram to boost contrast
3. Normalize to the range [-1,1] by apply the formula: `(pixel - 128)/128`
``` python
def normalize_set(X_src):
    X_aug = []
    for i in range(len(X_src)):
        img = X_src[i].astype(np.uint8)
        img_gry = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_contrast = cv2.equalizeHist(img_gry)
        img_out = (img_contrast.astype(np.float32) - 128)/128
        X_aug.append(img_contrast)
        
    return np.stack(X_aug,axis=0)
```
Here is a visualization of the normalized training set

![alt text][image8]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:----------------------|:----------------------------------------------| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 14x14x6 	    			|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 5x5x16 	    			|
| Flatten				| outputs 1x400									|
| Fully connected		| outputs 1x120									|
| RELU					|												|
| Dropout       		|           									|
| Fully connected		| outputs 1x84									|
| RELU					|												|
| Fully connected		| outputs 1x43									|
| Softmax				|           									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used a 2-step method to train the model.

In the first step, hyperparameters `BATCH_SIZE=128`, `learning_rate=0.00015`, `NUM_OF_EPOCH=35` and `keep_prob=0.9` are chosen to train the model with `AdamOptimzer`. It allows the model to find its way down the gradient faster with larger batch size and relatively larger learning rate. The trained model is saved as `lenet`.

In the second step, `lenet` is restored and is trained using hyperparameters `BATCH_SIZE=32`, `learning_rate=0.00009`, `NUM_OF_EPOCH=50` and `keep_prob=0.5`. Smaller batch size and learning rate allows the model to converge to the minimum in a finer steps. Introducing a larger dropout rate in this step also generalize the weights and biases to achieve a higher detection rate on unseen images. In fact, the second step is repeated multiple times with newly augmented training sets to feed more data into it.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of `98.2%`
* validation set accuracy of `96.4%`
* test set accuracy of `93.7%`

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

![LeNet Architecture](./writeup_imgs/lenet.png)

Source: Yan LeCun

A `Lenet-5 Max` architecture is chosen by the recommendation of this course as the baseline. But it struggles with achieving a higher accuracy while using different hyperparameters.  It is suggested that training the model multiple times using different hyperparameters is an effective way to achieve a higher accuracy model, so I tried the 2 steps method described in step `3` and the result is better. I also introduced dropout after the first fully connected layer to improve the robustness.

An interesting observation is that during the fine tuning phase, where dropout rate is increased from 10% to 50%, the training accuracy stays roughly the same but validation accuracy is increased gradually as it is trained more. It tells me that the dropout is effective to generalize the model to better classify unseen images.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

To answer both questions 1 and 2, here is graph that contains the original image found on the web, prediction label and the actual label.
And it achieves `82.9%` accuracy out of `35` images, so it mis-classifies `5` of the images. The mis-classified images are `20km/h speed limit` and `children crossing`. They both have very similar signs like a speed limit sign with different speed limit, and right-of-way in the next intersection. It probably needs more node in the network to capture those details.

![alt text][image9]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

A summary of the top5 predictions is shown below. 

![alt text][image10]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


