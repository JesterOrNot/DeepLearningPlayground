lib = __import__("lib")

print("Downloading Data...")
lib.download_data()
print("Organizing Data...")
lib.prep_data()
print("Removing Old Data...")
lib.remove_old_data()

train_data, test_data = lib.get_model_data()

model = lib.create_model(train_data[0], train_data[1], epochs=150, batch_size=10)

_, accuracy = model.evaluate(test_data[0], test_data[1], verbose=0)

predictions = (model.predict(test_data[1]) > 0.5).astype("int32")

print("Accuracy: %.2f" % (accuracy * 100))
