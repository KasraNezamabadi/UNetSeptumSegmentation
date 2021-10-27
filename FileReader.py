import csv
import os
import Global
import cv2
import numpy as np
if os.name == 'nt':
    import win32api, win32con


class DataReader(object):

    __csv_array = []

    __list_of_images = []
    __list_of_masks = []
    __list_of_thicknesses = []
    __list_of_distances = []

    def __init__(self, base_path):
        self.__base_path = base_path
        self.array_of_images = np.ndarray
        self.array_of_masks = np.ndarray
        self.array_of_thicknesses = np.ndarray

    def __load_csv(self):
        thickness_file = open(self.__base_path + '/thickness.csv')
        csv_reader = csv.reader(thickness_file, delimiter=',')
        count = 0
        for row in csv_reader:
            if count == 0:
                count += 1
                continue
            thickness = float(row[8])
            distance = float(row[5])
            self.__csv_array.append([str(row[0]), thickness, distance])
            count += 1

        thickness_file.close()

    def __find_thickness_and_distance(self, image_name):
        for row in self.__csv_array:
            if str(row[0]) == str(image_name):
                return True, row[1], row[2]
        return False, 0.0, 0.0

    def __folder_is_hidden(self, p):
        if os.name == 'nt':
            attribute = win32api.GetFileAttributes(p)
            return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
        else:
            return p.startswith('.')  # linux-osx

    def __read_files_from_disk(self):

        Global.log('Reading Files from Disk')
        self.__load_csv()
        path_image = self.__base_path + '/images'
        path_mask = self.__base_path + '/masks'

        list_of_image_names = [f for f in os.listdir(path_image) if not self.__folder_is_hidden(f)]
        list_of_mask_names = [f for f in os.listdir(path_mask) if not self.__folder_is_hidden(f)]

        assert len(list_of_image_names) == len(list_of_mask_names)

        index_of_image = 0

        for image_name in sorted(list_of_image_names):

            if ".png" not in image_name:
                Global.log('Not image file found. Skipping...', have_line=False)
                continue

            thickness_found, thickness, distance = self.__find_thickness_and_distance(image_name=image_name)
            if not thickness_found:
                continue

            image = cv2.imread(os.path.join(path_image, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = cv2.imread(os.path.join(path_mask, image_name))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            self.__list_of_images.append(image)
            self.__list_of_masks.append(mask)
            self.__list_of_thicknesses.append(thickness)
            self.__list_of_distances.append(distance)

            print('{0} -> {1}'.format(image_name, index_of_image))
            index_of_image += 1

        Global.log('Reading Files done! {0} tuples of (image, thickness, distance) read'.format(index_of_image),
                   have_line=False)

    def load_data_from_disk(self):

        self.__read_files_from_disk()

        number_of_samples = len(self.__list_of_images)
        image_width, image_height = np.array(self.__list_of_images[0]).shape[0], np.array(self.__list_of_images[0]).shape[1]
        if not image_width == image_height:
            if image_height > image_width:
                image_width = image_height
            else:
                image_height = image_width
        # self.array_of_images = np.ndarray((number_of_samples, image_width, image_height, 1), dtype=np.float32)
        # self.array_of_masks = np.ndarray((number_of_samples, image_width, image_height, 1), dtype=np.float32)
        # self.array_of_thicknesses = np.ndarray(number_of_samples, dtype=np.float32)


        #debug_num = 10
        self.array_of_images = np.ndarray((number_of_samples, image_width, image_height, 1), dtype=np.float32)
        self.array_of_masks = np.ndarray((number_of_samples, image_width, image_height, 1), dtype=np.float32)
        self.array_of_thicknesses = np.ndarray(number_of_samples, dtype=np.float32)

        Global.log('Converting Files into Numpy Format...')
        i = 0
        for image in self.__list_of_images:
            image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
            image = np.array(image)
            image = image[:, :, np.newaxis]
            self.array_of_images[i] = image
            mask = self.__list_of_masks[i]
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
            mask = np.array(mask)
            mask = mask[:, :, np.newaxis]
            self.array_of_masks[i] = mask
            self.array_of_thicknesses[i] = self.__list_of_thicknesses[i]
            # if i == debug_num:
            #     break
            i += 1

        Global.log('Done', have_line=False)









