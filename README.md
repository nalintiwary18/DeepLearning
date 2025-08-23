# ann

## Overview

This project is implemented in a Jupyter Notebook (`.ipynb`). It uses Python along with data science and machine learning libraries to achieve its task. Below is a breakdown of the technologies and explanations for each code block.

### Block 1: Code

**Purpose:** This block contains executable Python code.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
```


### Block 2: Code

**Purpose:** This block contains executable Python code.

```python
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)= fashion_mnist.load_data()
```


### Block 3: Code

**Purpose:** This block contains executable Python code.

```python
train_images.shape
```


### Block 4: Code

**Purpose:** This block contains executable Python code.

```python
train_images[0,23,23]
```


### Block 5: Code

**Purpose:** This block contains executable Python code.

```python
train_labels[:10]
```


### Block 6: Code

**Purpose:** This block contains executable Python code.

```python
class_names = ['T-shirt','Trouser','Pullover','Dress','coat','Sandal','shirt','Sneaker','bag','Ankle boot']
```


### Block 7: Code

**Purpose:** This block contains executable Python code.

```python
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()
```


### Block 8: Code

**Purpose:** This block contains executable Python code.

```python
train_images = train_images/255.0
test_images=test_images/255.0
```


### Block 9: Code

**Purpose:** This block contains executable Python code.

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```


### Block 10: Code

**Purpose:** This block contains executable Python code.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```


### Block 11: Code

**Purpose:** This block contains executable Python code.

```python
model.fit(train_images, train_labels, epochs=5)
```


### Block 12: Code

**Purpose:** This block contains executable Python code.

```python
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

print('Test accuracy:', test_acc)
```


### Block 13: Code

**Purpose:** This block contains executable Python code.

```python
predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[1])])
plt.figure()
plt.imshow(test_images[1])
plt.colorbar()
plt.grid(False)
plt.show()
```


### Block 14: Code

**Purpose:** This block contains executable Python code.

```python
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict
...```


### Block 15: Code

**Purpose:** This block contains executable Python code.

```python

```



---

# PlayGenerator

## Overview

This project is implemented in a Jupyter Notebook (`.ipynb`). It uses Python along with data science and machine learning libraries to achieve its task. Below is a breakdown of the technologies and explanations for each code block.

### Block 1: Code

**Purpose:** This block contains executable Python code.

```python
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np
```


### Block 2: Code

**Purpose:** This block contains executable Python code.

```python
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
```


### Block 3: Code

**Purpose:** This block contains executable Python code.

```python
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print ('Length of text: {} characters'.format(len(text)))
```


### Block 4: Code

**Purpose:** This block contains executable Python code.

```python
print(text[:250])
```


### Block 5: Code

**Purpose:** This block contains executable Python code.

```python
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
  return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)
```


### Block 6: Code

**Purpose:** This block contains executable Python code.

```python
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))
```


### Block 7: Code

**Purpose:** This block contains executable Python code.

```python
def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

print(int_to_text(text_as_int[:13]))
```


### Block 8: Code

**Purpose:** This block contains executable Python code.

```python
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
```


### Block 9: Code

**Purpose:** This block contains executable Python code.

```python
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
```


### Block 10: Code

**Purpose:** This block contains executable Python code.

```python
for x, y in dataset.take(2):
  print("\n\nEXAMPLE\n")
  print("INPUT")
  print(int_to_text(x))
  print("\nOUTPUT")
  print(int_to_text(y))
```


### Block 11: Code

**Purpose:** This block contains executable Python code.

```python
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
```


### Block 12: Code

**Purpose:** This block contains executable Python code.

```python
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                     
...```


### Block 13: Code

**Purpose:** This block contains executable Python code.

```python
for input_example_batch, target_example_batch in data.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
```


### Block 14: Code

**Purpose:** This block contains executable Python code.

```python
print(len(example_batch_predictions))
print(example_batch_predictions)
```


### Block 15: Code

**Purpose:** This block contains executable Python code.

```python
pred = example_batch_predictions[0]
print(len(pred))
print(pred)
```


### Block 16: Code

**Purpose:** This block contains executable Python code.

```python
time_pred = pred[0]
print(len(time_pred))
print(time_pred)
```


### Block 17: Code

**Purpose:** This block contains executable Python code.

```python
sampled_indices = tf.random.categorical(pred, num_samples=1)

sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
predicted_chars = int_to_text(sampled_indices)

predicted_chars
```


### Block 18: Code

**Purpose:** This block contains executable Python code.

```python
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
```


### Block 19: Code

**Purpose:** This block contains executable Python code.

```python
model.compile(optimizer='adam', loss=loss)
```


### Block 20: Code

**Purpose:** This block contains executable Python code.

```python
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
```


### Block 21: Code

**Purpose:** This block contains executable Python code.

```python
history = model.fit(data, epochs=50, callbacks=[checkpoint_callback])
```


### Block 22: Code

**Purpose:** This block contains executable Python code.

```python
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
```


### Block 23: Code

**Purpose:** This block contains executable Python code.

```python
latest_checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint_file:
    print(f"Found latest checkpoint: {latest_checkpoint_file}")
    model.build(tf.TensorShape([1, None])) # Build the model before loading weights
    model.load_weights(latest_checkpoint_file)
else:
    
...```


### Block 24: Code

**Purpose:** This block contains executable Python code.

```python
model.build(tf.TensorShape([1, None]))
model.load_weights("./training_checkpoints/ckpt_50.weights.h5") # Correct usage for .h5 files
```


### Block 25: Code

**Purpose:** This block contains executable Python code.

```python
def generate_text(model, start_string):

  num_generate = 800

  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = []

  temperature = 1.0


  model.layers[1].reset_states()

  for i in range(num_generate):
      predictions = model(inp
...```


### Block 26: Code

**Purpose:** This block contains executable Python code.

```python
inp = input("Type a starting string: ")
print(generate_text(model, inp))
```



---

# SentimentAnalysis

## Overview

This project is implemented in a Jupyter Notebook (`.ipynb`). It uses Python along with data science and machine learning libraries to achieve its task. Below is a breakdown of the technologies and explanations for each code block.

### Block 1: Code

**Purpose:** This block contains executable Python code.

```python
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)
```


### Block 2: Code

**Purpose:** This block contains executable Python code.

```python
train_data[1]
```


### Block 3: Code

**Purpose:** This block contains executable Python code.

```python
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)
```


### Block 4: Code

**Purpose:** This block contains executable Python code.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```


### Block 5: Code

**Purpose:** This block contains executable Python code.

```python
model.summary()
```


### Block 6: Code

**Purpose:** This block contains executable Python code.

```python
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)
```


### Block 7: Code

**Purpose:** This block contains executable Python code.

```python
results = model.evaluate(test_data, test_labels)
print(results)
```


### Block 8: Code

**Purpose:** This block contains executable Python code.

```python
word_index = imdb.get_word_index()

def encode_text(text):
  tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return sequence.pad_sequences([tokens], MAXLEN)[0]

text = "that movie was just amazing, so am
...```


### Block 9: Code

**Purpose:** This block contains executable Python code.

```python
reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
      if num != PAD:
        text += reverse_word_index[num] + " "

    return text[:-1]

print(decode_integers(encoded))
```


### Block 10: Code

**Purpose:** This block contains executable Python code.

```python

def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model.predict(pred)
  print(result[0])

positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
predict(positive_review)

ne
...```



---

