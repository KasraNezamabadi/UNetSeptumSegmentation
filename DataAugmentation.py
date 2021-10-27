import Global
import cv2
import numpy as np
import ElasticTransformation as ETA


class ElasticAugmentation(object):

    def __init__(self, array_of_images, array_of_masks):
        self.__array_of_images = array_of_images
        self.__array_of_masks = array_of_masks
        self.__list_of_images = []
        self.__list_of_masks = []
        self.array_augmented_images = []
        self.array_augmented_masks = []


    def find_anomely(self, image):
        temp = image
        temp = np.array(temp)
        if len(temp.shape) == 3:
            temp = temp[:, :, 0]

        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if temp[i][j] > 0 and temp[i][j] < 1:
                    return True
        return False

    def __perform_augmentation(self):
        i = 0
        for image in self.__array_of_images:
            image = np.array(image)
            mask = self.__array_of_masks[i]
            mask = np.array(mask)
            if len(image.shape) == 3:
                image = image[:, :, 0]

            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

            self.__list_of_images.append(image)
            self.__list_of_masks.append(mask)

            for alpha_elastic in range(60, 150, 30):
                for _ in range(5):
                    augmented_image, augmented_mask = ETA.elastic_transform(image=image, mask=mask, alpha=alpha_elastic, sigma=8)
                    self.__list_of_images.append(augmented_image)
                    self.__list_of_masks.append(augmented_mask)

            if i % 10 == 0:
                print('{0} images have been augmented!'.format(i))

            i += 1

    def augment_dataset(self, save_path=None):

        Global.log('Performing Morphological + Elastic Augmentation on Dataset...')

        self.__perform_augmentation()

        self.__list_of_images = np.array(self.__list_of_images)
        self.__list_of_masks = np.array(self.__list_of_masks)

        number_of_samples = self.__list_of_images.shape[0]
        image_width = self.__list_of_images.shape[1]
        image_height = self.__list_of_images.shape[2]

        self.array_augmented_images = np.ndarray((number_of_samples, image_width, image_height, 1), dtype=np.float32)
        self.array_augmented_masks = np.ndarray((number_of_samples, image_width, image_height, 1), dtype=np.float32)

        i = 0
        for image in self.__list_of_images:
            image = image[:, :, np.newaxis]
            self.array_augmented_images[i] = image
            i += 1

        i = 0
        for mask in self.__list_of_masks:
            mask = mask[:, :, np.newaxis]
            self.array_augmented_masks[i] = mask
            i += 1

    # def analyze_on_single_image(self, image, thickness):
    #     result_array = []
    #     result_thickness = []
    #
    #     image = np.array(image)
    #     if len(image.shape) == 3:
    #         image = image[:, :, 0]
    #
    #     temp_list_images, temp_list_thicknesses = MA.perform_morphological_augmentation(image=image,
    #                                                                                     thickness=thickness)
    #     self.__append_list(result_array, temp_list_images)
    #     self.__append_list(result_thickness, temp_list_thicknesses)
    #
    #     return result_array, result_thickness



