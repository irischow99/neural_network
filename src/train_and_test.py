import numpy as np
import os
from PIL import Image

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import iris_neural_net
net = iris_neural_net.Network([784, 30, 10])

#net.stochastic_gradient_descent(30, 3.0, 10, training_data, test_set=test_data)

def load_images(image_dir):
	arr = []
	files = os.listdir(image_dir)
	for file in files:
		file_name = os.path.join(image_dir, file)
		if ".png" in file_name:
			print(file_name[-5])
			img = Image.open(file_name).convert("L")
			img = img.resize((28, 28))
			imarr = np.array(img.getdata())
			imarr = imarr.reshape(784, 1)
			imarr -= 255
			imarr *= -1
			imarr = imarr / 255
			arr.append(imarr)
	return arr

print('Expected Output:')
test_images = load_images('../images')

print('Actual Output:')
for image in test_images:
	print(np.argmax(net.forward_prop(image)))