from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Prep Data
dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
# Get the first 8 fields from each row
x = dataset[:,0:8]
# Get Last element of each row
y = dataset[:,8]

model = Sequential([
  Dense(12, input_dim=8, activation='relu'),
  Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=150, batch_size=10, verbose=0)

_, accuracy = model.evaluate(x, y, verbose=0)

# make class predictions with the model
predictions = (model.predict(x) > 0.5).astype("int32")
# summarize the first 5 cases
for i in range(len(x)):
	print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))

print('Accuracy: %.2f' % (accuracy*100))
