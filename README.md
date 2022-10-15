[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/sample_human_output.png "Sample Output Human"
[image3]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image4]: ./images/vgg16_model_draw.png "VGG16 Model Figure"

## Table of Contents

1. [Project Overview](#overview)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview <a name="overview"></a>

For people, it is a not difficult task or rather trivial to see and understand the images in front of them. Computer Vision (CV) is a multi-disciplinary field that focuses on enabling computers to do so from digital images, videos and other visual input. A human can easily tell if it is either a dog or a human face, and even recognize a face/breed that they have only seen once before. How can we render the same or even more capabilities to computers?

While it remains unsolved, CV has made significant progress over these decades. Among them, deep learning methods such as Convolutional Neural Networks (CNN) are achieving promising results on challenging problems, which include image classification, object recognition, face detection, and so forth.

Taking advantage of such progress, this project aims at constructing a deep learning algorithm to classify images of dogs and human faces according to the dog's breed. The algorithm development is intended to be used as part of a web or mobile app, which accepts any user-supplied image as input and provides an estimate of the dog's breed name for either dog or human-face image.

![Sample Output][image1]
![Sample Output Human][image2]

## Project Motivation<a name="motivation"></a>
The objective of this project is to develop an algorithm that can take an image as input and return the prediction of the dog’s breed. If a human image is detected, it will provide an estimate of the dog breed (this will be fun to know your dog cousin!). Therefore, the tasks to address this problem are as follows:

1.	Develop Human-face detector
2.	Develop Dog detector
3.	Create a CNN to classify dog breeds
4.	Develop and test an algorithm for the whole process

For the development of CNN to classify dog breeds, transfer learning will be utilized based on the pre-computed bottleneck features.

## File Descriptions <a name="files"></a>
Jupyter Notebook [dog_app_submit.ipynb](dog_app_submit.ipynb) contains all the codes for the above task. Markdown cells were used to describe the questions and findings of the data analysis.

There is also a word document [Report](dog_app_REPORT.docx) that discuss this project in detail and serves as the manuscript for the blog article.

## Instructions <a name="instructions"></a>

1. Clone the repository and navigate to the downloaded folder.
```
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.

4. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`):
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`):
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

7. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```

8. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```

9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__:
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__:
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment.
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app_submit.ipynb
```
5. [Results](#results)
