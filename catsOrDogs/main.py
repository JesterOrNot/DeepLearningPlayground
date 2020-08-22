lib = __import__("lib")

lib.download_data()

photos, labels = lib.get_formatted_data()


model = lib.create_model(photos, labels, epochs=150, batch_size=10)

_, accuracy = model.evaluate(photos, labels, verbose=0)

# make class predictions with the model
# predictions = (model.predict(photos) > 0.5).astype("int32")

print("Accuracy: %.2f" % (accuracy * 100))
