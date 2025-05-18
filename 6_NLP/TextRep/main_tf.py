import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
# In this tutorial, we will be training a lot of models. In order to use GPU memory cautiously,
# we will set tensorflow option to grow GPU memory allocation when required.

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


print(tf.__version__)
dataset = tfds.load('ag_news_subset')

ds_train = dataset['train']
ds_test = dataset['test']

print(f"Length of train dataset = {len(ds_train)}")
print(f"Length of test dataset = {len(ds_test)}")


classes = ['World', 'Sports', 'Business', 'Sci/Tech']

for i,x in zip(range(5), ds_train):
    print(f"{x['label']} ({classes[x['label']]}) -> {x['title']} {x['description']}")


vocab_size = 50000
vectorizer = keras.layers.TextVectorization(max_tokens=vocab_size)
vectorizer.adapt(ds_train.take(500).map(lambda x: x['title']+ ' '+x['description']))

vocab = vectorizer.get_vocabulary()
vocab_size = len(vocab)
print(vocab[:10])
print(f"Length of vocabulary: {vocab_size}")

vectorizer('I love to play with my words')