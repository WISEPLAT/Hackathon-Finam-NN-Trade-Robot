"""
    В этом коде мы обучаем нейросеть (НС), для входа используем timeframe_0,
    для выхода timeframe_1. Модели НС сохраняются в папку NN/_models.
    После обучения, лучшую модель НС сохраняем вручную в папку NN_winner.

    Логи работы сохранены в файлах:
    - 3_Training and Validation Accuracy and Loss.jpg - график Training and Validation Accuracy and Loss
    - 3_results_of_training_neural_network.txt - процесс обучения нейросети логи с экрана
    Итак в процессе этого обучения лучше всего себя показала модель на эпохе 27:
Epoch 26/40
449/449 [==============================] - 25s 55ms/step - loss: 0.0402 - accuracy: 0.9871 - val_loss: 0.2765 - val_accuracy: 0.9312
Epoch 27/40
449/449 [==============================] - 25s 55ms/step - loss: 0.0386 - accuracy: 0.9875 - val_loss: 0.1685 - val_accuracy: 0.9563
Epoch 28/40
449/449 [==============================] - 25s 55ms/step - loss: 0.0370 - accuracy: 0.9875 - val_loss: 0.2051 - val_accuracy: 0.9491
    Выбираем 27-ю к помещаем её в папку NN_winner под именем cnn_Open.hdf5.

    Автор: Олег Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

exit(777)  # для запрета запуска кода, иначе перепишет результаты

import functions
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow import config
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Rescaling
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint

from my_config.trade_config import Config  # Файл конфигурации торгового робота

print("Num GPUs Available: ", len(config.list_physical_devices('GPU')))

if __name__ == '__main__':  # Точка входа при запуске этого скрипта

    # перенаправлять ли вывод с консоли в файл
    functions.start_redirect_output_from_screen_to_file(False, filename="3_results_of_training_neural_network.txt")

    # ------------------------------------------------------------------------------------------------------------------

    timeframe_0 = Config.timeframe_0  # таймфрейм для обучения нейросети - вход - для картинок
    draw_size = Config.draw_size  # размер стороны квадратной картинки

    # ================================================================================================================

    cur_run_folder = os.path.abspath(os.getcwd())  # текущий каталог
    data_dir = os.path.join(os.path.join(cur_run_folder, "NN"), f"training_dataset_{timeframe_0}")  # каталог с данными
    num_classes = 2  # всего классов
    epochs = 40  # Количество эпох
    batch_size = 10  # Размер мини-выборки
    img_height, img_width = draw_size, draw_size  # размер картинок
    input_shape = (img_height, img_width, 3)  # размерность картинки

    # # Первый тип модели
    # model = Sequential()
    # model.add(Rescaling(1. / 255))
    # model.add(Conv2D(64, (3, 3), input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    # model.add(Conv2D(32, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    # model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes))
    # model.add(Activation('sigmoid'))
    # # version with Gradient descent (with momentum) optimizer
    # model.compile(
    #     optimizer=keras.optimizers.SGD(),
    #     loss=keras.losses.SparseCategoricalCrossentropy(),
    #     metrics=['accuracy']
    # )

    # Второй тип модели
    model = keras.Sequential([
        keras.layers.Rescaling(1. / 255),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes)
    ])
    # version with Adam optimization is a stochastic gradient descent method
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # model.summary()

    # тренировочный набор
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        # seed=123,
        shuffle=False,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # набор для валидации
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        # seed=123,
        shuffle=False,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # # нормализация есть прямо в модели
    # normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    # train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # для записи моделей
    callbacks = [ModelCheckpoint(functions.join_paths([cur_run_folder, "NN", "_models", 'cnn_Open{epoch:1d}.hdf5'])),
                 # keras.callbacks.EarlyStopping(monitor='loss', patience=10),
                 ]

    # запуск процесса обучения
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # графики потерь и точности на обучающих и проверочных наборах
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("3_Training and Validation Accuracy and Loss.png", dpi=150)
    plt.show()

    # ================================================================================================================

    # остановка перенаправления вывода с консоли в файл
    functions.stop_redirect_output_from_screen_to_file()
