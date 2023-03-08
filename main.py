import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os 
from PIL import Image
import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model
import time


# Loading the data 
IMG_DIMS = (100, 100)
def load_dataset(path):
    dataset = []
    files = os.listdir(path)
    
    for i in files:
        if i.endswith(".jpg"):
            img = Image.open(f"{path}/{i}", 'r')
            img = img.resize(IMG_DIMS)
            pix_val = list(img.getdata())
            dataset.append(pix_val)

    dataset = np.array(dataset)
    return dataset


original_images = load_dataset("dataset/me/me")
original_images = np.reshape(original_images, (original_images.shape[0], 100, 100, 3))

other_images = load_dataset("dataset/other/other")
other_images = np.reshape(other_images, (other_images.shape[0], 100, 100, 3))


# Data normalization from [0, 1]
original_images = original_images / 255.0
other_images = other_images / 255.0

print(original_images.shape)

# plt.imshow(original_images[0])
# plt.show()


# Custom Hyper-parameters and layers
class Dense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        # Using term kernal as the term 'weights' cannot be used
        self.kernal = self.add_weight(shape=[batch_input_shape[-1], self.units], 
                                      initializer='glorot_normal', 
                                      name='weights')
        
        self.bias = self.add_weight(shape=[self.units], 
                                    initializer='zeros', 
                                    name='bias')
        
        super().build(batch_input_shape)

    def call(self, X):
        return X @ self.kernal + self.bias
    

class Flatten(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, X):
        ''' Note :- This does not work on a local machine use Google Colab instead idk
                    why some tensorflow bug or version error.
        '''
        output = tf.reshape(a, [1, -1])
        return output 

def relu(X):
    return tf.math.maximum(X, 0)



CODING_DIM = 20
# Original image Encoder 
''' 
Currently using keras.layers.Dense instead of custom Dense because of initializer 
try finding better one than glorot_normal maybe lecunn_normal may work. Also check bias initializer
'''
input_data = keras.layers.Input(shape=[100, 100, 3])
encoder = keras.Sequential([
    # keras.layers.Flatten(),
    Flatten(), 
    keras.layers.Dense(1200, activation=relu),
    keras.layers.Dense(700, activation=relu),
    keras.layers.Dense(150, activation=relu), 
    keras.layers.Dense(CODING_DIM, activation=relu), 
])

decoder = keras.Sequential([
    keras.layers.Dense(150, activation=relu),
    keras.layers.Dense(700, activation=relu),
    keras.layers.Dense(1200, activation=relu),
    keras.layers.Dense(100 * 100 * 3),
    keras.layers.Reshape([100, 100, 3])
])

encode = encoder(input_data)
decode = decoder(encode)

model = keras.Model(input_data, decode)
# model.compile(optimizer='nadam', loss='mse', metrics=['accuracy'])
# model.fit(original_images, original_images, epochs=5, batch_size=10)
# model.save("my-face-model/")


# model.fit(other_images, other_images, epochs=5, batch_size=10)
# model.save("other-face-model/")

# Custom training loop 
optimizer = keras.optimizers.Adam()
loss = keras.losses.MeanSquaredError()

@tf.function 
def step(x, orig):
    with tf.GradientTape() as model_gradients:
        preds = model(x, training=True)
        model_loss = loss(preds, orig)
        print(f"Loss: {model_loss}")

    gradients = model_gradients.gradient(model_loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables))



def train(x, y, epochs, name):
    for epoch in range(epochs):
        start = time.time()  
        step(x, y)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print("-----------------------------")

    # Saving the mpdel in the end
    if epoch == epochs:
        model.save(f"{name}/")
    


train(original_images, original_images, 10, "my-face-model")
train(other_images, other_images, 10, "other-face-model")


# Testing img autoencoder
model_1 = keras.models.load_model("my-face-model/")
model_2 = keras.models.load_model("other-face-model/")
test_images_1 = original_images[0:10]
test_images_2 = other_images[0:10]
prediction_1 = model_1.predict(test_images_1)
prediction_2 = model_2.predict(test_images_2)
plt.imshow(prediction_1[0])
plt.show()
plt.imshow(prediction_2[0])
plt.show()


# Loading in the models and getting predictions 
model_original = load_model("my-face-model/")
other_model = load_model("other-face-model/")



# Predictions 
print(model_original.layers)

encoder_a = keras.Model(
    model_original.layers[1].input, model_original.layers[1].output)

decoder_a = keras.Model(
    model_original.layers[2].input, model_original.layers[2].output)

encoder_b = keras.Model(
    other_model.layers[1].input, other_model.layers[1].output)

decoder_b = keras.Model(
    other_model.layers[2].input, other_model.layers[2].output)



input_test = encoder_a.predict(np.array([original_images[0]]))
output_test = decoder_b.predict(input_test)

plt.imshow(output_test[0])
plt.show()
