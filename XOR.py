import numpy as np
import random
from Perceptron import Perceptron

class XOR:
	def __init__(self):
		self.training_examples = [[0.0,0.0],
								  [0.0,1.0],
								  [1.0,0.0],
								  [1.0,1.0]]
		self.not_training_examples = [[0],
									  [1]]
		self.not_target_labels = [1,0]
		self.or_target_labels = [0,1,1,1]

		self.and_target_labels = [0,0,0,1]

		self.nand_target_labels = [1,1,1,0]

		self.or_perceptron = None
		self.not_perceptron = None
		self.nand_perceptron = None
		self.and_perceptron = None

	def create_perceptrons(self):
		self.or_perceptron = Perceptron(2, bias=-1,float_threshold=0)
		self.not_perceptron = Perceptron(1,bias =0.5,float_threshold=0)
		self.nand_perceptron = Perceptron(2, bias=3,float_threshold=0)
		self.and_perceptron = Perceptron(2,bias=-3,float_threshold=0)
		print("Working up to here")

	def generate_training_examples(self):
		examples = 0
		input_ranges = [(-0.25,0.25),(0.75,1.25)]
		while examples < 1000:
			input1_range = input_ranges[np.random.choice([0,1],p=[0.5,0.5])]
			input2_range = input_ranges[np.random.choice([0,1],p=[0.5,0.5])]

			input_1 = round(random.uniform(input1_range[0],input1_range[1]),2)
			input_2 = round(random.uniform(input2_range[0],input2_range[1]),2)
			#print([input_1,input_2])
			self.training_examples.append([input_1,input_2])
			if input_1 > 0.75 and input_2 > 0.75:
				self.and_target_labels.append(1)
			else:
				self.and_target_labels.append(0)

			if input_1 > 0.75 or input_2 > 0.75:
				self.or_target_labels.append(1)
			else:
				self.or_target_labels.append(0)
			examples += 1

	def train_perceptrons(self):
		iterations = 0

		while iterations < 100000:
			self.or_perceptron.train(self.training_examples,self.or_target_labels,0.5)
			self.not_perceptron.train(self.not_training_examples,self.not_target_labels,0.5)
			self.nand_perceptron.train(self.training_examples,self.nand_target_labels,0.5)
			self.and_perceptron.train(self.training_examples,self.and_target_labels,0.2)
			iterations+=1

		print("Final Weight")
		print(self.or_perceptron.weights)

	def construct_network(self):
		print("Done")

	def start_XOR(self):
		#create perceptrons set biases to -1 for now
		self.create_perceptrons()
		self.generate_training_examples()
		print("There are " + str(len(self.training_examples)) + " examples")
		print("There are " + str(len(self.and_target_labels)) + " AND training labels")
		print("There are " + str(len(self.or_target_labels)) + " OR training labels")
		for i in range(len(self.training_examples)):
			print(str(self.training_examples[i]) + " " + str(self.and_target_labels[i]))
		return
		self.train_perceptrons()
		print("Constructing Network")
		self.construct_network()
		while True:
			try:
				user_input = input("Please enter two inputs: ").split()
				if user_input[0] == "exit":
					print("Exiting...")
					break
				x1 = float(user_input[0])
				x2 = float(user_input[1])

				user_input = [x1,x2]
				print(user_input)
				#or_output = self.or_perceptron.activate(user_input)
				#print("Or Output = "+ str(or_output))
				print("And Output")
				print(self.and_perceptron.activate(user_input))
				#nand_output = self.nand_perceptron.activate(user_input)
				#print("Nand Output = " + str(nand_output))
				#xor_output = self.and_perceptron.activate([or_output,nand_output])
				#print("XOR Output")
				#print(xor_output)
				#print("Not Output")
				#print(self.not_perceptron.activate([x1]))

			except ValueError:
				print("Please enter a valid pair of numbers")


class driverClass:
	def main():
		xor_nn = XOR()
		xor_nn.start_XOR()
	if __name__ == "__main__":
		main()