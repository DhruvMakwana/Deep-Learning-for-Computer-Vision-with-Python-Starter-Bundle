# importing libraries
from sklearn.preprocessing import LabelBinarizer
from dhruv.nn.conv import minivggnet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to best model weights file")
args = vars(ap.parse_args())

print("[Info] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

print("[Info] compiling network...")
opt = SGD(lr=0.01, decay=0.01/5, momentum=0.9, nesterov=True)
model = minivggnet.MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

checkpoint = ModelCheckpoint(args["weights"], monitor="val-loss", save_best_only=True, verbose=1)
callbacks = [checkpoint]

print("[Info] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=5, callbacks=callbacks, verbose=2)
