import numpy as np
from Perceptron import Perceptron

class XOR:
	def __init__(self):
		self.training_examples = [[0.0,0.0],
								  [0.0,1.0],
								  [1.0,0.0],
								  [1.0,1.0]]
		self.or_target_labels = [0,1,1,1]

		self.and_target_labels = [0,0,0,1]

		self.nand_target_labels = [1,1,1,0]

		self.or_perceptron = None
		self.nand_perceptron = None
		self.and_perceptron = None

	def create_perceptrons(self):
		self.or_perceptron = Perceptron(2, bias=-1)
		self.nand_perceptron = Perceptron(2, bias=-1)
		self.and_perceptron = Perceptron(2,bias=-1)
		print("Working up to here")

	def train_perceptrons(self):
		iterations = 0

		while iterations < 50:
			self.or_perceptron.train(self.training_examples,self.or_target_labels,0.2)
			self.nand_perceptron.train(self.training_examples,self.nand_target_labels,0.2)
			self.and_perceptron.train(self.training_examples,self.and_target_labels,0.2)
			iterations+=1

	def construct_network(self):
		print("Done")

	def start_XOR(self):
		#create perceptrons set biases to -1 for now
		self.create_perceptrons()
		self.train_perceptrons()
		print("Constructing Network")
		self.construct_network()
		while True:
			try:
				user_input = input("Please enter two inputs: ").split()
				x1 = float(user_input[0])
				x2 = float(user_input[1])
				print(str(x1) + "," + str(x2))
			except ValueError:
				print("Please enter a valid pair of numbers")


class driverClass:
	def main():
		xor_nn = XOR()
		xor_nn.start_XOR()
	if __name__ == "__main__":
		main()