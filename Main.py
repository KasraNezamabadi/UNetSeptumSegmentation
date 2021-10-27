from __future__ import print_function
from skimage.io import imsave
import numpy as np
from keras.callbacks import ModelCheckpoint
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import os
import datetime

import DataSet
import Global
import UNet
#import OutputVisualizer
from DataSetOOP import DataSourceManager
from keras import backend as KerasBackend
import pandas as pd
import tensorflow as tf
import cv2


os.environ['KMP_DUPLICATE_LIB_OK']='True'


date = datetime.datetime.today().strftime('%Y-%m-%d')

learning_rate = 0.001
batch_size = 16
number_of_epochs = 10
validation_split = 0.2


def plot(history):
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Model Dice Coefficient')
    plt.ylabel('Dice Coef')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('result/charts/{0}-model_dice.png'.format(date), bbox_inches='tight')


def evaluation(model, test_image, test_mask, test_thickness, fold_number):

    Global.log('Restoring Trained Weights...')
    model.load_weights('weights/weights.h5')

    Global.log('Evaluation Initiated. \nEvaluating Network on Test Data...')
    eval = model.evaluate(x=test_image, y=test_mask, batch_size=batch_size)
    print('Dice Coefficient on Testset: {0}'.format(eval[1]))
    Global.log('Running Network on Test Data...')
    imgs_mask_test = model.predict(test_image, verbose=1)

    dice_coefficient_array = []
    thickness_array = []
    i = 0
    for mask_predicted in imgs_mask_test:
        print(np.max(mask_predicted))
        cv2.imwrite('UNetMaskResults/' + 'fold-' + str(fold_number) + '-' + str(i) + '.png', mask_predicted)
        mask_actual = test_mask[i]
        thickness = test_thickness[i]
        dice = UNet.dice_coef(y_pred=mask_predicted, y_true=mask_actual)
        dice_coefficient_array.append(tf.keras.backend.eval(dice))
        thickness_array.append(thickness)
        i += 1

    compare = pd.DataFrame(data={'Dice Coefficient': dice_coefficient_array,
                                 'Thickness': thickness_array})
    compare.to_csv('UNet' + '-' + date + '-fold-' + str(fold_number) + '.csv')


def run():
    ds = DataSourceManager(base_path='Data2', save_path='Data2/dataset')
    number_of_folds = ds.number_of_folds

    for fold_index in range(number_of_folds):
        Global.log('Fold {0}'.format(fold_index + 1))
        train_images, train_masks, test_images, test_masks, test_thicknesses = ds.load_dataset(
            fold_number=fold_index + 1)

        model = UNet.get_model(number_of_image_rows=np.array(train_images[0]).shape[0],
                               number_of_image_columns=np.array(train_images[0]).shape[1],
                               learning_rate=learning_rate)

        model_checkpoint = ModelCheckpoint('weights/weights.h5',
                                           monitor='val_loss',
                                           save_best_only=True)

        Global.log('Fitting Model Initiated...')
        history = model.fit(x=train_images, y=train_masks,
                            batch_size=batch_size,
                            nb_epoch=number_of_epochs,
                            verbose=1,
                            shuffle=True,
                            validation_split=validation_split,
                            callbacks=[model_checkpoint])

        evaluation(model=model,
                   test_image=test_images,
                   test_mask=test_masks,
                   test_thickness=test_thicknesses,
                   fold_number=fold_index + 1)


if __name__ == '__main__':
    run()
