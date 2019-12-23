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

"""
if not os.path.exists('updated_images'):
    print("New directory created")
    os.makedirs('updated_images')

if not os.path.exists('faces'):
    print("New directory created")
    os.makedirs('faces')

base_dir = os.path.dirname(__file__)

for file in os.listdir(base_dir + '/images'):
    file_name, file_extension = os.path.splitext(file)
    if file_extension in ['.png', '.jpg']:
        print("Image path: {}".format(base_dir + '/images/' + file))

    image = cv2.imread(base_dir + '/images/' + file)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()

    print(detections.shape[2])
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, save it as a separate file
        print(confidence)
        if confidence > 0.5:
            frame = image[startY:endY, startX:endX]
            cv2.imwrite(base_dir + '/faces/' + str(i) + '_' + file, frame)
"""


# класс подготовки последовательности изображений
class Sequence:
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
    def create_sequence(self, test, epoch_size):
        random.seed()
        self.images = []
        self.labels = []
        k = 0
        train_coefficient = 1
        if test:
            train_coefficient = 0.1

        """self.seq_num_folders_1.clear()
        self.seq_num_folders_2.clear()
        self.seq_num_image_1.clear()
        self.seq_num_image_2.clear()
        """
        print('function create is started')
        a = round((1 - train_coefficient) * len(self.folder_list))
        b = round(len(self.folder_list) * 0.9)
        if test:
            b = len(self.folder_list) - 1
        print('_ _ _ _ _ _ _ _ _ _')
        while k <= round(epoch_size * train_coefficient)-1:
            """
            self.seq_num_folders_1.append(random.randint(0, len(self.folder_list)))
            self.seq_num_folders_2.append(random.randint(0, len(self.folder_list)))
            self.seq_num_image_1.append(random.randint(0, 790))
            self.seq_num_image_2.append(random.randint(0, 790))
            """

            # номера папок откуда берутся лица, поочередное совпадение

            ident_1_folder = random.randint(a, b)
            if k % 2 == 0:
                ident_2_folder = random.randint(a, b)

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
            flag = True
            label = 0
            if ident_1_folder == ident_2_folder:
                label = 1

            # вытаскивание изображений и нормализация в список фреймов(лиц), если в массиве одно лицо,
            # то изображение не идет в обучающее множество
            frames_1 = normalization(image_1_folder[ident_1_image], ("/celeb/" + self.folder_list[ident_1_folder]))
            frames_2 = normalization(image_2_folder[ident_2_image], ("/celeb/" + self.folder_list[ident_2_folder]))
            if len(frames_1) != 1 or len(frames_2) != 1:
                flag = False

            if flag:
                # self.images_label.append([np.concatenate((frames_1[0], frames_2[0]), axis=1), label])
                self.images.append(np.concatenate((frames_1[0], frames_2[0]), axis=1))

                # cv2.imshow(str(label), self.images[len(self.images)-1])
                # cv2.waitKey(0)

                self.labels.append(label)
                # cv2.imwrite(base_dir + '/clear_data/' + str(label) + '/' + image_1_folder[ident_1_image], image_)
                if k % round(round(epoch_size * train_coefficient) / 10) == 0:
                    print('= ', end='')
                k += 1
        self.index += 1
        print(' ')
        images_ = np.squeeze(self.images)
        labels_ = np.squeeze(self.labels)
        return shuffle(round(epoch_size * train_coefficient), images_, labels_)


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


def shuffle(len, array_image, array_label):
    random.seed()
    x = [[i] for i in range(len)]
    random.shuffle(x)
    temp_image = np.copy(array_image)
    temp_label = np.copy(array_label)
    for i in range(len):
        temp_image[i] = array_image[x[i]]
        temp_label[i] = array_label[x[i]]
    return [temp_image, temp_label]

"""
# тест файла
seq = Sequence()
sequence = seq.create_sequence(False)
print('op')
for i in range(100):
    print(sequence[1][i])
    cv2.imshow(str(sequence[1][i]), sequence[0][i])
    cv2.waitKey(0)"""