"""
# Librerías
"""

import math, re, os, zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.applications import EfficientNetV2B0
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

"""# Variables y ubicaciones"""

# ==== Configuración de variables ====

classes = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102

# tfrecords jpeg 224x224 path
ds_path = "/LUSTRE/home/rn_lcc_02/sebastian_browarski/clasificacion_flores/tfrecords-jpeg-224x224"

train_files = tf.io.gfile.glob(f"{ds_path}/train/*.tfrec")
val_files = tf.io.gfile.glob(f"{ds_path}/val/*.tfrec")
test_files = tf.io.gfile.glob(f"{ds_path}/test/*.tfrec")

"""# Función para visualización"""

def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()
    #plt.savefig('cm.jpg')
    plt.close()

"""# Funciones Preprocesamiento"""

# ==== Funciones para manejo de datos obtenidas del Kaggle ====

AUTO = tf.data.experimental.AUTOTUNE
image_size = [224, 224]
batch_size = 32

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.efficientnet.preprocess_input(image)  # Normaliza para EfficientNet
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return image, label

def get_training_dataset():
    dataset = load_dataset(train_files, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(val_files, labeled=True, ordered=ordered)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(test_files, labeled=False, ordered=ordered)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

num_training_images = count_data_items(train_files)
num_val_images = count_data_items(val_files)
num_test_images = count_data_items(test_files)
steps_per_epoch = num_training_images // batch_size
val_steps = -(-num_val_images // batch_size)
test_steps = -(-num_test_images // batch_size)
print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(num_training_images, num_val_images, num_test_images))


train = get_training_dataset()
val = get_validation_dataset()
test = get_test_dataset()

"""# Modelo"""

# ==== Transferencia de aprendizaje ====
# Modelo utilizado: EfficientNetV2 (B0)
# Nos ofrece una arquitectura moderna optimizada para datasets medianos, y B0 (7.1M params) es rápido y suficiente.

# Entrada
inputs = Input(shape=(224, 224, 3))

# Modelo base
base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg')
base_model.trainable = False

# Capas personalizadas
x = base_model.output
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(classes), activation='softmax')(x)

# Crear el modelo funcional
model = Model(inputs=inputs, outputs=outputs)

# Compilar y entrenar como lo hacías
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

history = model.fit(
    train,
    epochs=15,
    steps_per_epoch=steps_per_epoch,
    validation_data=val,
    validation_steps=val_steps,
)

# Visualización antes de fine-tuning
# Grafica de accuracy
def plot_hist(hist):
    plt.plot(hist.history["sparse_categorical_accuracy"])
    plt.plot(hist.history["val_sparse_categorical_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.ylim((0,1.1))
    plt.grid()
    plt.show()
    #plt.savefig('accuracy.jpg')
    plt.close()

plot_hist(history)

# Grafica de loss
def plot_hist_loss(hist):
    plt.plot(hist.history["loss"],'.r')
    plt.plot(hist.history["val_loss"],'*b')
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    #plt.ylim((0,1))
    plt.grid()
    plt.show()
    #plt.savefig('loss.jpg')
    plt.close()

plot_hist_loss(history)

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

history_finetune = model.fit(
    train,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_data=val,
    validation_steps=val_steps
)

# Visualización despues de fine-tuning
# Grafica de accuracy
def plot_hist(hist):
    plt.plot(hist.history["sparse_categorical_accuracy"])
    plt.plot(hist.history["val_sparse_categorical_accuracy"])
    plt.title("Model Accuracy (fine-tune)")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.ylim((0,1.1))
    plt.grid()
    plt.show()
    #plt.savefig('accuracy_fine_tuning.jpg')
    plt.close()

plot_hist(history_finetune)

# Grafica de loss
def plot_hist_loss(hist):
    plt.plot(hist.history["loss"],'.r')
    plt.plot(hist.history["val_loss"],'*b')
    plt.title("Model Loss (fine-tune)")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    #plt.ylim((0,1))
    plt.grid()
    plt.show()
    #plt.savefig('loss_fine_tuning.jpg')
    plt.close()

plot_hist_loss(history_finetune)

# Test dataset
test_ds = get_test_dataset(ordered=True)

# Separar imágenes e ids
test_images = test_ds.map(lambda image, idnum: image)
test_ids = test_ds.map(lambda image, idnum: idnum)

# Predicciones
test_pred_probs = model.predict(test_images, steps=test_steps)
test_preds = np.argmax(test_pred_probs, axis=-1)

# ids del conjunto de prueba
ids = []
for batch in test_ids:
    for id in batch.numpy():
        ids.append(id.decode("utf-8"))

# DataFrame con las predicciones
pred_df = pd.DataFrame({
    'id': ids,
    'label': test_preds
})

# Archivo output_submission.csv con los valores reales
output_submission = pd.read_csv("/LUSTRE/home/rn_lcc_02/sebastian_browarski/clasificacion_flores/output_submission.csv")

# Merge de DataFrames con etiquetas verdaderas y predicciones
merged = pd.merge(output_submission, pred_df, on='id', how='inner')

# Comparar las predicciones con las etiquetas verdaderas
y_true = merged['label_x']  # Etiquetas verdaderas
y_pred = merged['label_y']  # Predicciones

print(confusion_matrix(y_true, y_pred, labels=range(len(classes))))
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

display_confusion_matrix(cm, score=None, precision=None, recall=None)

cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
cm_norm = (cm.T / cm.sum(axis=1)).T

display_confusion_matrix(cm_norm, score=None, precision=None, recall=None)

model.save('model.keras') 
model.export('model')
