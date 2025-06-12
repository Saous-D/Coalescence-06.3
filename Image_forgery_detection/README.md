# Image Forgery Detection using CNN

Detects the authenticity of an image using Error Level Analysis and Convolutional Neural Networks.

## Introduction

### Error Level Analysis
 > Error Level Analysis is an image feature extraction technique. JPEG is a lossy compression format and hence the pixels in the images stored in that format tend to be downsampled from which they are captured initially. When an authentic image is loaded into an image editing software, tampered and then stored again in JPG format, the pixels where the image has been tampered will have distinct compression artifacts relative to the adjacent pixels. These compression artifacts can be analysed to detect the presence of image tampering.  

### Convolutional Neural Network
 > A Convolutional Neural Network(ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance(learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.

## Setup and Usage

1. Upgrade pip.
2. Install the following py modules

	* ```dwave-ocean-sdk```
    * ```keras```
    * ```numpy```
    * ```pillow```
    * ```PyQt5```
	* ```PyQt5-tools```
	* ```scipy```
	* ```tensorflow```
	
It is desirable to create a virtual environment in the local machine before installing the listed python packages.
In order to do this, in the chosen folder, one should type ```python -m venv <pyenvName>```, followed by ```source <pyenvName>/bin/activate```, where ```<pyenvName>``` is the name of the python virtual environment chosen by the user.	
3. Clone this repository to your local machine.
4. Run the ```ui.py``` file, using the command ```python ui.py```.
5. Browse an image from your local machine and test it, by clicking on the "Test" button.

## Dataset
[Casia dataset - Kaggle](https://www.kaggle.com/sophatvathana/casia-dataset)
* Total Images : 11129
* Authentic images : 8144
* Forged images : 2985

## Contributors
<a href="https://github.com/0xsp/image-forgery-detection/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=0xsp/image-forgery-detection" />
  <img src="https://github.com/TBG-QbitSoft/Coalescence/network/dependencies" />
</a>

## License
MIT

## Reference  
[FotoForensics](https://fotoforensics.com/)
