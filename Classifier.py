import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
import random
import sys
import matplotlib.pyplot as plt
from Perceptron import Perceptron

class Classifier:
	def __init__(self):
		self.epochs = 10
		self.number_of_classes = 10
		self.learning_rate = 0.01
		self.file_path = sys.argv[1]
		self.transform = None
		self.training_set = None
		self.validation_set = None
		self.data_loader = None
		self.neural_network = None
		self.input_layer_size = 28 * 28
		self.hidden_layer1_size = 128
		self.hidden_layer2_size = 64


	def define_transforms(self):
		self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor(),])
		print(self.transform)

	def create_training_set(self):
		self.training_set = torchvision.datasets.ImageFolder(root="./trainingSet/trainingSet/",transform=self.transform)
		self.data_loader = torch.utils.data.DataLoader(self.training_set,batch_size=64,shuffle=True)

		data_iterator = iter(self.data_loader)
		images,labels = data_iterator.next()

		print(images.shape)
		#plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
		print("DONE")

	def create_neural_network(self):
		input_to_hidden1 = nn.Linear(self.input_layer_size,self.hidden_layer1_size)
		hidden1_to_hidden2 = nn.Linear(self.hidden_layer1_size,self.hidden_layer2_size)
		hidden2_to_output = nn.Linear(self.hidden_layer2_size,self.number_of_classes)

		self.neural_network = nn.Sequential(input_to_hidden1,nn.ReLU(),
											hidden1_to_hidden2,nn.ReLU(),
											hidden2_to_output,nn.LogSoftmax(dim=1))
		print(self.neural_network)

	def start_classification(self):
		print(2)
		self.define_transforms()
		self.create_training_set()
		self.create_neural_network()
		print(self.file_path)

class driverClass:
	def main():
		digit_classifier = Classifier()
		digit_classifier.start_classification()
	if __name__ == "__main__":
		main()