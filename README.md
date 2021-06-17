<h1 align="center"> Artificial Neural Networks - The XOR Problem/ Classifying Handwritten Digits</h1>

<div align="center" >
  <img src="https://img.shields.io/badge/made%20by-Zongo%20Maqutu-blue?style=for-the-badge&labelColor=20232a" />
  <img src="https://img.shields.io/badge/Python 3.8.5-20232a?style=for-the-badge&logo=python&labelColor=20232a" />
  <img src="https://img.shields.io/badge/Numpy-20232a?style=for-the-badge&logo=numpy&labelColor=162e16" />
  <img src="https://img.shields.io/badge/Pycharm-20232a?style=for-the-badge&logo=pycharm&labelColor=517a8a" />
</div>

## Table of Contents
* [Project Setup](#ProjectSetup)
* [Libraries Used](#libraries)
* [Future Scope](#FutureScope)
## Description 
*These two programs perform different tasks. The first program (XOR.py) trains a multi-layered network of perceptrons to make accurate output predictions given noisy input data corresponding to 1 or 0 (on or off).*

*The second program (Classifier.py) creates and trains a neural network to accurately classify a handwritten digit. Digits are passed in as JPG images of 28x28 pixels*
  

## Project setup  
To run this project clone this repository in a folder on your local machine.
We first need to build our virtual environment and install a list of 
libraries our program needs to run. To do this, open a terminal in the root directory and run the following commands

```
make install       // installs program dependencies
```


Next we need to activate our virual environment. To do this run the following commands

```
source venv/bin/activate       // Activates our virtual environment
```

Now we can run either one of our programs (they are in the same directory). First make sure that your training data is pasted in the right folder. Make sure that the folder trainingSet/trainingSet/ is present and has all labeled inputs. Once that is done, run these commands and you will be provided with sample inputs to your file for each of the 
two algorithms

```
python XOR.py       // runs the XOR.py program
```
or
```
python Classifier.py           //runs the image classifier
```
Alternatively you can run each program program with your own arguments that follow the pattern 

To exit the virtual environment run:

```
deactivate       // runs the program
```
### Libraries Used
* Numpy
* Pytorch
* Torchvision
* PIL


## Future Scope
* add the logic from this project to the pathfinding-visualizer project in my repo which can be found at
https://github.com/zmaqutu/3D-Pathfinding-Visualizer

* Use JavaScript/React to create better animations

<p align="center">Made with ❤️ with Pycharm and vim</p>





