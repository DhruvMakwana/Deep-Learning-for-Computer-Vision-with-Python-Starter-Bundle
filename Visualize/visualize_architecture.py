#importing libraries
from dhruv.nn.conv import lenet
from keras.utils import plot_model

model = lenet.LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="lenet.png", show_shapes=True)