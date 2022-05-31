import random
import keras_tuner

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers


def list_split(listA, n):
    for x in range(0, len(listA), n):
        every_chunk = listA[x: n+x]

        if len(every_chunk) < n:
            every_chunk = every_chunk + \
                [None for y in range(n-len(every_chunk))]
        yield every_chunk

def load_data():
    bad = np.load("bad_move_2.npy")
    good = np.load("good_move_2.npy")
    z = 100
    bad_split = np.array_split(bad,z)
    good_split = np.array_split(good, z)




    good_norm = []
    bad_norm = []
    label_g = [0] * (len(good_split))
    labe_n = [1] * (len(bad_split))
    for i in range(len(bad_split)):
        if (len(bad_split[i]) != 996):
            bad_split[i] = bad_split[i][:-1]

        norm_b = np.linalg.norm(bad_split[i])
        bad_norm.append(bad_split[i] / norm_b)

    for i in range(len(good_split)):
        if(len(good_split[i])!=996):
            good_split[i] = good_split[i][:-1]

        norm_g = np.linalg.norm(good_split[i])
        good_norm.append(good_split[i] / norm_g)




    plt.plot(good_norm[25])
    plt.plot(bad_norm[25])
    plt.show()

    data = np.concatenate((good_norm, bad_norm))
    data_label = np.concatenate((label_g, labe_n))
    c = list(zip(data, data_label))
    random.shuffle(c)
    data, data_label = zip(*c)




    return data,data_label
data,label_data = load_data()


model = keras.Sequential(
    [
        layers.Input(shape=(data.shape[1], data.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.MaxPooling1D(
            pool_size=3, strides=None, padding="valid"
        ),
        layers.Dropout(rate=0.4),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)




epochs = 200
batch_size = 32

data = np.asarray(data)
label_data = np.asarray(label_data)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model_2.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    )
]
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    data,
    label_data,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.3,
    verbose=1,
)



model = keras.models.load_model("best_model.h5")


metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()