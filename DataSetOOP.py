import cv2
import numpy as np
import Global
from FileReader import DataReader
import UNet
import tensorflow as tf
from DataAugmentation import ElasticAugmentation


class DataSourceManager(object):

    def __init__(self, base_path, save_path):
        self.number_of_folds = 5
        self.__base_path = base_path
        self.__save_path = save_path
        self.data_reader = DataReader(base_path=base_path)

    def save_dataset(self, perform_augmentation=False):
        self.data_reader.load_data_from_disk()
        array_of_images = self.data_reader.array_of_images
        array_of_masks = self.data_reader.array_of_masks
        array_of_thickness = self.data_reader.array_of_thicknesses
        number_of_samples = array_of_images.shape[0]
        number_of_test_samples = int(number_of_samples / self.number_of_folds)

        for i in range(1, self.number_of_folds+1):

            test_indecies = list(range((i-1)*number_of_test_samples, i * number_of_test_samples))

            array_of_images_test = np.array(array_of_images[test_indecies, :, :, :])
            array_of_masks_test = np.array(array_of_masks[test_indecies, :, :, :])
            array_of_thickness_test = np.array(array_of_thickness[test_indecies])

            array_of_images_train = np.array(np.delete(array_of_images, test_indecies, axis=0))
            array_of_masks_train = np.array(np.delete(array_of_masks, test_indecies, axis=0))

            if perform_augmentation:
                ea = ElasticAugmentation(array_of_images=array_of_images_train,
                                         array_of_masks=array_of_masks)

                ea.augment_dataset()
                array_of_images_train = ea.array_augmented_images
                array_of_masks_train = ea.array_augmented_masks

            Global.log('Saving Train and Test Sets for Fold {0}'.format(i))

            np.save(self.__save_path + '/images_train' + '_fold_' + str(i) + '.npy', array_of_images_train)
            np.save(self.__save_path + '/masks_train' + '_fold_' + str(i) + '.npy', array_of_masks_train)

            np.save(self.__save_path + '/images_test' + '_fold_' + str(i) + '.npy', array_of_images_test)
            np.save(self.__save_path + '/masks_test' + '_fold_' + str(i) + '.npy', array_of_masks_test)
            np.save(self.__save_path + '/thicknesses_test' + '_fold_' + str(i) + '.npy', array_of_thickness_test)

        print('Saving Done!')

    def load_dataset(self, fold_number, resize_rate=1.0):

        Global.log('Loading Fold {0} Dataset from {1}'.format(fold_number, self.__save_path))
        list_of_images_train = np.array(np.load(self.__save_path + '/images_train' + '_fold_' + str(fold_number) + '.npy'))
        list_of_masks_train = np.array(
            np.load(self.__save_path + '/masks_train' + '_fold_' + str(fold_number) + '.npy'))

        list_of_images_test = np.array(np.load(self.__save_path + '/images_test' + '_fold_' + str(fold_number) + '.npy'))
        list_of_masks_test = np.array(
            np.load(self.__save_path + '/masks_test' + '_fold_' + str(fold_number) + '.npy'))
        list_of_thicknesses_test = np.array(np.load(self.__save_path + '/thicknesses_test' + '_fold_' + str(fold_number) + '.npy'))

        if not resize_rate == 1.0:
            list_of_images_train, _, _ = self.__preprocess(list_of_images_train, resize_rate=resize_rate)
            list_of_masks_train, _, _ = self.__preprocess(list_of_masks_train, resize_rate=resize_rate)
            list_of_images_test, _, _ = self.__preprocess(list_of_images_test, resize_rate=resize_rate)
            list_of_masks_test, _, _ = self.__preprocess(list_of_masks_test, resize_rate=resize_rate)

        size = 128
        trimed_list_of_images_train = []
        trimed_list_of_masks_train = []
        trimed_list_of_images_test = []
        trimed_list_of_masks_test = []

        for image in list_of_images_train:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
            image = np.array(image)
            image = image[:, :, np.newaxis]
            max_pixel = float(np.max(image))
            image = np.float32(image) / max_pixel
            trimed_list_of_images_train.append(image)

        for mask in list_of_masks_train:
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_CUBIC)
            _, mask = cv2.threshold(src=mask, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
            mask = np.array(mask)
            mask = mask[:, :, np.newaxis]
            mask = np.float32(mask) / float(np.max(mask))
            trimed_list_of_masks_train.append(mask)

        for image in list_of_images_test:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
            image = np.array(image)
            image = image[:, :, np.newaxis]
            max_pixel = float(np.max(image))
            image = np.float32(image) / max_pixel
            trimed_list_of_images_test.append(image)

        for mask in list_of_masks_test:
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_CUBIC)
            _, mask = cv2.threshold(src=mask, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
            mask = np.array(mask)
            mask = mask[:, :, np.newaxis]
            mask = np.float32(mask) / float(np.max(mask))
            trimed_list_of_masks_test.append(mask)

        number_of_samples = len(trimed_list_of_images_train)
        array_images_train = np.ndarray((len(trimed_list_of_images_train), size, size, 1), dtype=np.float32)
        array_masks_train = np.ndarray((len(trimed_list_of_images_train), size, size, 1), dtype=np.float32)
        array_images_test = np.ndarray((len(trimed_list_of_images_test), size, size, 1), dtype=np.float32)
        array_masks_test = np.ndarray((len(trimed_list_of_images_test), size, size, 1), dtype=np.float32)

        i = 0
        for image in trimed_list_of_images_train:
            array_images_train[i] = image
            i += 1

        i = 0
        for image in trimed_list_of_masks_train:
            array_masks_train[i] = image
            i += 1

        i = 0
        for image in trimed_list_of_images_test:
            array_images_test[i] = image
            i += 1

        i = 0
        for image in trimed_list_of_masks_test:
            array_masks_test[i] = image
            i += 1

        return array_images_train,\
               array_masks_train,\
               array_images_test,\
               array_masks_test, \
               list_of_thicknesses_test

    # private function
    def __preprocess(self, list_of_images, resize_rate):
        sample_image = list_of_images[0]
        sample_image = np.array(sample_image)
        sample_image = cv2.resize(sample_image, (0, 0), fx=resize_rate, fy=resize_rate, interpolation=cv2.INTER_CUBIC)
        sample_image = np.array(sample_image)

        resized_image_width = sample_image.shape[0]
        resized_image_height = sample_image.shape[1]

        list_of_images_result = np.ndarray((len(list_of_images), resized_image_width, resized_image_height, 1), dtype=np.float32)

        i = 0
        for image in list_of_images:
            resized_image = cv2.resize(image, (0, 0), fx=resize_rate, fy=resize_rate, interpolation=cv2.INTER_CUBIC)
            resized_image = np.array(resized_image)
            resized_image = resized_image[:, :, np.newaxis]
            list_of_images_result[i] = resized_image
            i += 1

        return list_of_images_result, resized_image_width, resized_image_height


if __name__ == '__main__':

    ds = DataSourceManager(base_path='Data2', save_path='Data2/dataset')
    ds.save_dataset(perform_augmentation=True)
    #ds.load_dataset(fold_number=1)


