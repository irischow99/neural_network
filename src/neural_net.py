import random
import numpy as np

class Network(object):
	def __init__(self, widths):
		self.widths = widths
		self.layers = len(widths)
		self.biases = [np.random.randn(y, 1) for y in widths[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(widths[:-1], widths[1:])]

	"""Given an input, calculate the output of the neural network"""
	def forward_prop(self, input):
		for i in range(len(self.biases)):
			bias = self.biases[i]
			weight = self.weights[i]
			input = sigmoid(np.dot(weight, input) + bias)
		return input

	"""Run SGD for 'epochs' number of epochs with eta 'learning_rate', splitting the training set into chunks of size 'batch_size'"""
	def stochastic_gradient_descent(self, epochs, learning_rate, batch_size, training_set, test_set=None):
		if test_set:
			test_len = len(test_set)
		n = len(training_set)
		for epoch in range(epochs):
			random.shuffle(training_set)
			batches = []
			for i in range(0, n, batch_size):
				batches.append(training_set[i:i+batch_size])
			for batch in batches:
				self.update_biases_and_weights(batch, learning_rate)
			if test_set:
				print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_set), test_len))
			else:
				print("Epoch {0}".format(epoch))

	
	"""Update the values of the biases and the weights of the neural network using SGD"""
	def update_biases_and_weights(self, batch, learning_rate):
		learning_rate = learning_rate / len(batch)
		new_biases = [np.zeros(bias.shape) for bias in self.biases]
		new_weights = [np.zeros(weight.shape) for weight in self.weights]
		for input, expected_output in batch:
			bias_gradients, weight_gradients = self.backprop(input, expected_output)
			new_biases = np.add(new_biases, bias_gradients)
			new_weights = np.add(new_weights, weight_gradients)
		self.weights = np.add(self.weights, (-learning_rate) * new_weights)
		self.biases = np.add(self.biases, (-learning_rate) * new_biases)


	"""Backprop algorithm adapted from http://neuralnetworksanddeeplearning.com/chap1.html, 
	"Neural Networks and Deep Learning" by Michael Nielson, October 2018"""
	def backprop(self, x, y):
		"""Return a tuple of the gradient for the cost function of the neural network.  ``nabla_b`` and 
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar to ``self.biases`` and ``self.weights``."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		activation = x
		activations = [x]
		zs = []
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = (activations[-1] - y) * sigmoid(zs[-1]) * (1 - sigmoid(zs[-1]))
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		for l in range(2, self.layers):
			z = zs[-l]
			sp = sigmoid(z) * (1 - sigmoid(z))
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)


	"""Evaluate how many test inputs the neural network gets the correct answer"""
	def evaluate(self, test_set):
		sum = 0
		for x, y in test_set:
			if np.argmax(self.forward_prop(x)) == y:
				sum += 1
		return sum

"""Perform sigmoid operation on input n"""
def sigmoid(n):
	return 1.0 / (1.0 + np.exp(-n))

"""## License

MIT License

Copyright (c) 2012-2018 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""
