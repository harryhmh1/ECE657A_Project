import tensorflow as tf
import keras as kr
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
from keras.utils import plot_model
import matplotlib.pyplot as plt
import sys
from keras.layers.core import Dense, Dropout, Activation

stdout_backup = sys.stdout
# log_file = open("ece657Alogportland2.log", "w")
# sys.stdout = log_file
le = LabelEncoder()
raw_data = pd.read_csv("portland2.csv", names=["date_time", "hour", "temperature", "humidity", "pressure", "wind_speed", "weather"])

# y = raw_data.loc[119:, ["weather"]]
# X = raw_data.loc[:42630, ["temperature", "humidity", "pressure", "wind_speed"]]
y = raw_data[["weather"]]
X = raw_data[["hour", "temperature", "humidity", "pressure", "wind_speed"]]
# X = raw_data.loc[:42630, ["temperature", "humidity", "pressure", "wind_speed"]]

X = stats.zscore(X)
y = le.fit_transform(y)
y = kr.utils.to_categorical(y, num_classes=None, dtype='float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

init = kr.initializers.glorot_uniform(seed=1)
simple_adam = kr.optimizers.Adam()
model = kr.models.Sequential()
model.add(kr.layers.Dense(units=200, input_dim=5, kernel_initializer=init, activation='linear'))
model.add(kr.layers.Dense(units=100, kernel_initializer=init, activation='linear'))
model.add(Dropout(0.2))
model.add(kr.layers.Dense(units=90, kernel_initializer=init, activation='relu'))
model.add(kr.layers.Dense(units=80, kernel_initializer=init, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(kr.layers.Dense(units=70,  kernel_initializer=init, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(kr.layers.Dense(units=60,  kernel_initializer=init, activation='relu'))
model.add(kr.layers.Dense(units=50, kernel_initializer=init, activation='relu'))
model.add(kr.layers.Dense(units=40, kernel_initializer=init, activation='linear'))
model.add(kr.layers.Dense(units=30, kernel_initializer=init, activation='linear'))
model.add(kr.layers.Dense(units=20, kernel_initializer=init, activation='linear'))
model.add(kr.layers.Dense(units=12, kernel_initializer=init, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
# categorical_crossentropy
# binary_crossentropy
b_size = 50
max_epochs = 100
print("Starting training ")
h = model.fit(X_train, y_train, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1, validation_split=0.33)
print("Training finished \n")
eval = model.evaluate(X_test, y_test, verbose=0)
print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1] * 100))
# log_file.close()

plot_model(model, to_file='model_portland2.png', show_shapes=True)
# print(h.history.keys())


plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("acc_consequence_portland2")
plt.show()

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("loss_consequence_portland2")
plt.show()