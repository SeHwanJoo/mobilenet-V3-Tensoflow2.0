import os
import json
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from dataset_util import load_images, build_optimizer


def train():
    with open('config/config.json', 'r') as f:
        cfg = json.load(f)

    save_dir = cfg['save_dir']
    shape = (int(cfg['height']), int(cfg['width']))
    n_class = int(cfg['class_number'])
    batch = int(cfg['batch'])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    train_images, train_labels, test_images, test_labels = load_images()

    if cfg['model'] == 'large':
        from model.mobilenet_v3_large import MobileNetV3_Large
        model = MobileNetV3_Large(train_images[0].shape, n_class).build(shape=shape)
    if cfg['model'] == 'small':
        from model.mobilenet_v3_small import MobileNetV3_Small
        model = MobileNetV3_Small(train_images[0].shape, n_class).build(shape=shape)


    optimizer = build_optimizer(learning_rate=float(cfg['learning_rate']), momentum=0.9)
    # earlystop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_dir, '{}_weights.h5'.format(cfg['model'])),
                                 monitor='val_acc', save_best_only=True, save_weights_only=True)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()



    # data augmentation
    datagen1 = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen1.fit(train_images)

    hist = model.fit_generator(
        datagen1.flow(train_images, train_labels, batch_size=batch),
        validation_data=(test_images, test_labels),
        steps_per_epoch=train_images.shape[0] // batch,
        epochs=cfg['epochs'],
        callbacks=[checkpoint])

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(os.path.join(save_dir, 'hist.csv'), encoding='utf-8', index=False)
    # model.save_weights(os.path.join(save_dir, '{}_weights.h5'.format(cfg['model'])))


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[4], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[4], True)
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[4],
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        except RuntimeError as e:
            # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
            print(e)

    train()
