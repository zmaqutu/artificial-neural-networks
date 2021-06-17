import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
import random
import sys
from PIL import Image
from torchvision.transforms import ToTensor
from torch.autograd import Variable
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
		self.loss_function = None
		self.optimizer = None
		self.criteria = nn.NLLLoss()
		self.user_input_loader = None


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
		print(labels.shape)

	def create_neural_network(self):
		input_to_hidden1 = nn.Linear(self.input_layer_size,self.hidden_layer1_size)
		hidden1_to_hidden2 = nn.Linear(self.hidden_layer1_size,self.hidden_layer2_size)
		hidden2_to_output = nn.Linear(self.hidden_layer2_size,self.number_of_classes)

		self.neural_network = nn.Sequential(input_to_hidden1,nn.ReLU(),
											hidden1_to_hidden2,nn.ReLU(),
											hidden2_to_output,nn.LogSoftmax(dim=1))
		print(self.neural_network)

	def define_loss_function(self):
		images,labels = next(iter(self.data_loader))
		images = images.view(images.shape[0],-1)

		probabilities = self.neural_network(images)
		self.loss_function = self.criteria(probabilities,labels)

	def train_neural_net(self):
		self.optimizer = optim.SGD(self.neural_network.parameters(),lr=0.003,momentum=0.9)

		for epoch in range(1):
			current_loss = 0
			for images,labels in self.data_loader:
				images = images.view(images.shape[0],-1)

				self.optimizer.zero_grad()
				output = self.neural_network(images)
				self.loss_function = self.criteria(output,labels)

				self.loss_function.backward()
				self.optimizer.step()

				current_loss = current_loss + self.loss_function.item()
		#print(self.neural_network[0].weight.grad)

	def load_image(self, image_name):
		image = Image.open(image_name)
		image = self.transform(image).float()
		image = image.unsqueeze_(0)
		image = Variable(image)
		return image

	def classify_image(self):
		#images,labels = next(iter())
		while True:
			file_path = input("Please enter a file path: ")
			if file_path == "EXIT":
				break
			image = self.load_image(file_path)
			predicted_output  = torch.argmax(self.neural_network(image.view(-1,784)))
			print(predicted_output)

	def start_classification(self):
		print(2)
		self.define_transforms()
		self.create_training_set()
		self.create_neural_network()
		self.define_loss_function()
		print('Before backward pass: \n', self.neural_network[0].weight.grad)
		self.loss_function.backward()
		print('After backward pass: \n', self.neural_network[0].weight.grad)
		self.train_neural_net()
		self.classify_image()
		print(self.file_path)

class driverClass:
	def main():
		digit_classifier = Classifier()
		digit_classifier.start_classification()
	if __name__ == "__main__":
		main()