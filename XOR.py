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

	def start_XOR(self):
		#create perceptrons set biases to -1 for now
		or_perceptron = Perceptron(2,bias=-1)
		nand_perceptron = Perceptron(2,bias=-1)
		and_perceptron = Perceptron(2,bias=-1)

class driverClass:
	def main():
		xor_nn = XOR()
		xor_nn.start_XOR()
	if __name__ == "__main__":
		main()