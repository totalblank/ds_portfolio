import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
def _():
    # imports
    import marimo as mo
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    from matplotlib import pyplot as plt
    import pandas as pd

    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.datasets import imdb
    return keras, layers, mnist, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""# Chapter 5""")
    return


@app.cell
def _(mnist, np):
    (train_images, train_labels), _ = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255

    train_images_with_noise_channels = np.concatenate(
        [train_images, np.random.random((len(train_images), 784))], axis=1
    )

    train_images_with_zero_channels = np.concatenate(
        [train_images, np.zeros((len(train_images), 784))], axis=1
    )
    return (
        train_images,
        train_images_with_noise_channels,
        train_images_with_zero_channels,
        train_labels,
    )


@app.cell
def _(keras, layers):
    def get_model():
        model = keras.Sequential(
            [
                layers.Dense(512, activation="relu"),
                layers.Dense(10, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
    return (get_model,)


@app.cell
def _(get_model, train_images_with_noise_channels, train_labels):
    model1 = get_model()
    history_noise = model1.fit(
        train_images_with_noise_channels,
        train_labels,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=0
    )
    return (history_noise,)


@app.cell
def _(get_model, train_images_with_zero_channels, train_labels):
    model2 = get_model()
    history_zeros = model2.fit(
        train_images_with_zero_channels,
        train_labels,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=0
    )
    return (history_zeros,)


@app.cell
def _(history_noise, history_zeros, plt):
    val_acc_noise = history_noise.history["val_accuracy"]
    val_acc_zeros = history_zeros.history["val_accuracy"]
    epochs = range(1, 11)

    plt.plot(
        epochs,
        val_acc_noise,
        "b-",
        label="Validation accuracy with noise channels",
    )

    plt.plot(
        epochs,
        val_acc_zeros,
        "b--",
        label="Validation accuracy with zeros channels",
    )

    plt.title("Effect of noise channles on validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md(r"""Despite the data holding the same information in both cases, the validation accuracy of the model with noise channels ends up about one percentage point lower $\rightarrow$ purely through the influence of spurious correlations. The more noise channels you add, the furthur accuracy will degrade.""")
    return


@app.cell
def _(mo):
    mo.md(r"""## The nature of generalization in deep learning""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Deep learning models can be trained to fit anything, as long as they have enough representation power. With this logic, deep learning models shouldn't generalize at all. But the nature nature of generalization in deep learning has rather little to do with deep learning models themselves, adn much to do with the structure of information in the real world. Check out [The Manifold Hypothesis](https://en.wikipedia.org/wiki/Manifold_hypothesis), which basically states that all samples in the valid subspace are *connected* by smooth paths that run through the supspace. This means that if you take two random MNIST digits A and B, there exists a sequence of "intermediate" images that morph A into B, such that two consecutive digits are very close to each other.

    More generally, the *manifold hypothesis* posits that all natural data lies on a low-dimensional manifold within the high-dimensional space where it is encoded.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## High learning rate""")
    return


@app.cell
def _(keras, layers, train_images, train_labels):
    high_lr_model = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    high_lr_model.compile(
        optimizer=keras.optimizers.RMSprop(2.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    high_lr_history = high_lr_model.fit(
        train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2, verbose=0
    )
    return high_lr_history, high_lr_model


@app.cell
def _(high_lr_history, mo):
    mo.md(f"""The above model has accuracy {round(max(high_lr_history.history['accuracy']), 2)}""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Low learning rate model""")
    return


@app.cell
def _(high_lr_model, keras, layers, train_images, train_labels):
    low_lr_model = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    low_lr_model.compile(
        optimizer=keras.optimizers.RMSprop(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    low_lr_history = high_lr_model.fit(
        train_images,
        train_labels,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=0
    )
    return (low_lr_history,)


@app.cell
def _(low_lr_history, mo):
    mo.md(f"""The above model has accuracy {round(max(low_lr_history.history['accuracy']), 2)}""")
    return


@app.cell
def _(keras, layers, plt, train_images, train_labels):
    big_model = keras.Sequential([
        layers.Dense(96, activation="relu"),
        layers.Dense(96, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    big_model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_large_model = big_model.fit(
        train_images,
        train_labels,
        epochs=20,
        batch_size=128,
        validation_split=0.2,
        verbose=0
    )

    plt.plot(
        range(1,21),
        history_large_model.history['val_loss'],
        "b--",
        label="Validation loss"
    )
    plt.title("Overfits!")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
