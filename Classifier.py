import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
import random
import sys
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

	def define_transforms(self):
		self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)),])
		print(self.transform)

	def create_training_set(self):
		self.training_set = torchvision.datasets.ImageFolder(root="./trainingSet/trainingSet/",transform=self.transform)
		self.data_loader = torch.utils.data.DataLoader(self.training_set,batch_size=64,shuffle=True)

		data_iterator = iter(self.data_loader)
		images,labels = data_iterator.next()

		print(images.shape)

	def start_classification(self):
		print(2)
		self.define_transforms()
		self.create_training_set()
		print(self.file_path)

class driverClass:
	def main():
		digit_classifier = Classifier()
		digit_classifier.start_classification()
	if __name__ == "__main__":
		main()