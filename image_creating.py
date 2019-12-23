import os
import cv2
import numpy as np
import random

IMAGE_SIZE = 30
base_dir = os.path.dirname(__file__)

# для поиска лица на изображении
# используется в функции normalization
prototxt_path = os.path.join(base_dir + '/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + '/weights.caffemodel')
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


class Images:
    global EPOCHS
    global base_dir
    global IMAGE_SIZE
    index = 0
    """
    image_1_data = []
    image_2_data = []
    seq_num_folders_1 = []
    seq_num_folders_2 = []
    seq_num_image_1 = []
    seq_num_image_2 = []
    """

    def __init__(self):
        # список списков со склеинными изображениями и метки

        self.folder_list = os.listdir(base_dir + '/celeb/')

        return

    # функция возвращает список пар (в массиве) изображение-метка с лицами из разных папок
    # так как отбрасывает варианты с несколькими лицами, то для проверки стоит закидывать только изображения
    # с уже нормализованными лицами
    def save_images(self, epoch_size, flag_):
        random.seed()
        self.images = []
        self.labels = []
        k = 0

        a = 0
        b = len(self.folder_list) - 1
        max_ = max_number_search(flag_)
        print('function create is started')
        while k <= epoch_size -1:

            # номера папок откуда берутся лица, поочередное совпадение

            ident_1_folder = random.randint(a, b)
            if flag_ == 0:
                ident_2_folder = random.randint(a, b)
                if ident_2_folder == ident_1_folder:
                    if ident_2_folder <= 1:
                        ident_2_folder += 1
                    else:
                        ident_2_folder -= 1
            else:
                ident_2_folder = ident_1_folder



            # списки с фото из соответствующих папок
            image_1_folder = os.listdir(base_dir + '/celeb/' + self.folder_list[ident_1_folder])
            image_2_folder = os.listdir(base_dir +
                                        '/celeb/' +
                                        self.folder_list[ident_2_folder])

            # случайный номер фоток
            ident_1_image = random.randint(0, len(image_1_folder) - 1)
            ident_2_image = random.randint(0, len(image_2_folder) - 1)

            # флаг для наличия нескольких лиц на изображении, метка для наличия совпадения лиц
            flag_size = True

            # вытаскивание изображений и нормализация в список фреймов(лиц), если в массиве одно лицо,
            # то изображение не идет в обучающее множество
            frames_1 = normalization(image_1_folder[ident_1_image], ("/celeb/" + self.folder_list[ident_1_folder]))
            frames_2 = normalization(image_2_folder[ident_2_image], ("/celeb/" + self.folder_list[ident_2_folder]))
            if len(frames_1) != 1 or len(frames_2) != 1:
                flag_size = False

            if flag_size:
                number = max_ + k + 1
                cv2.imwrite(base_dir + '/data/' + str(flag_) + '/' + str(number) + '.jpg',np.concatenate((frames_1[0], frames_2[0]), axis=1))
                if epoch_size > 10:
                    if k % round(epoch_size / 10) == 0:
                        print('= ', end='')
                k += 1
        self.index += 1
        print('')
        return 1


def normalization(file_name, folder_name):
    dsize = (IMAGE_SIZE, IMAGE_SIZE)
    frames = []
    file_name, file_extension = os.path.splitext(file_name)
    image = cv2.imread(base_dir + folder_name + "/" + file_name + ".jpg")

    if file_extension != '.jpg' or isinstance(image, type(None)):
        # print(base_dir + folder_name + "/" + file_name + ".jpg")
        # print(type(image))
        return [1, 1]

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.8, save it as a separate file
        if confidence > 0.8:

            src = image[startY:endY, startX:endX]
            if type(src) != 'NoneType' and src.shape[0]>0 and src.shape[1]:
                frame = cv2.resize(src, dsize)
                frames.append(frame)
    return frames


def max_number_search(flag):
    image_folder_max = os.listdir(base_dir + '/data/' + str(flag))
    if len(image_folder_max) == 0:
        return 0
    temp = []
    for image in image_folder_max:
        temp.append(int(image[:-4]))
    print(temp)
    return max(temp)


def main():

    im = Images()

    print('align number of pictures? (1 = yes)')
    align_input = int(input())
    if align_input == 1:
        image_1_folder_align = os.listdir(base_dir + '/data/1')
        image_0_folder_align = os.listdir(base_dir + '/data/0')
        len_0 = len(image_0_folder_align)
        len_1 = len(image_1_folder_align)
        print(len_0, len_1)
        if len_1 != len_0:
            if len_1 > len_0:
                im.save_images(len_1 - len_0, 0)
            else:
                im.save_images(len_0 - len_1, 1)

        image_1_folder_align = os.listdir(base_dir + '/data/1')
        image_0_folder_align = os.listdir(base_dir + '/data/0')
        len_0 = len(image_0_folder_align)
        len_1 = len(image_1_folder_align)
        if len_1 == len_0:
            print('aligned')

    print('How many create? (n)')
    number_input = int(input())
    print('two types or not? (<0 = yes, 0 = 0, 1 = 1')
    types_input = int(input())
    if types_input < 0:
        im.save_images(number_input, 0)
        im.save_images(number_input, 1)
    else:
        if types_input == 0:
            im.save_images(number_input, 0)
        if types_input == 1:
            im.save_images(number_input, 1)

    print('0-s:', len(os.listdir(base_dir + '/data/0')), '\n', '1-s:', len(os.listdir(base_dir + '/data/1')))


class FolderImages:
    def __init__(self):
        return

    def upload_images_train(self, number_):
        print('uploading started')
        k = 0
        image_list = []
        label_list = []
        folder_1_list = os.listdir(base_dir + '/data/1')
        folder_0_list = os.listdir(base_dir + '/data/0')
        while k <= number_ - 1:
            print(k, end='\r')
            if k % 2 == 0:
                image_list.append(cv2.imread(base_dir + '/data/0/' + folder_0_list[round(k / 2)]))
                label_list.append(0)
            else:
                image_list.append(cv2.imread(base_dir + '/data/1/' + folder_1_list[round((k - 1) / 2)]))
                label_list.append(1)
            k += 1

        print('uploading ended')
        return [np.squeeze(image_list), np.squeeze(label_list)]

    def upload_images_test(self, number_):
        print('uploading started')
        k = 0
        image_list = []
        label_list = []
        folder_1_list = os.listdir(base_dir + '/data/1')
        len_1 = len(folder_1_list)
        folder_0_list = os.listdir(base_dir + '/data/0')
        len_0 = len(folder_0_list)
        while k <= number_ - 1:
            if k % 2 == 0:
                image_list.append(cv2.imread(base_dir + '/data/0/' + folder_0_list[len_0 - round(k / 2) - 1]))
                label_list.append(0)
            else:
                image_list.append(cv2.imread(base_dir + '/data/1/' + folder_1_list[len_1 - round((k - 1) / 2) - 1]))
                label_list.append(1)
            k += 1

        print('uploading ended')
        return [np.squeeze(image_list), np.squeeze(label_list)]

