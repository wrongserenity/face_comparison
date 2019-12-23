import tensorflow
import os
from tensorflow import keras

from tensorflow.keras.layers import Dense, Dropout, Flatten, Concatenate, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model

from image_add import Sequence
from generator import FolderImages
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_SIZE = 30
base_dir = 'D:/python/recognition/saved_models/'

# creating
def create_images(n):
    """
    start = time.time()
    seq = Sequence()
    sequence1 = seq.create_sequence(False, n)
    train_images, train_labels = sequence1[0], sequence1[1]
    end = time.time()
    print('trainset created', end - start)

    start = time.time()
    sequence2 = seq.create_sequence(True, n)
    test_images, test_labels = sequence2[0], sequence2[1]
    end = time.time()
    print('testset created', end - start)

    start = time.time()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    end = time.time()
    print('images normalized', end - start)
    """
    start = time.time()
    images__ = FolderImages()
    images_train = images__.upload_images_train(round(n * 0.9))
    train_images, train_labels = images_train[0], images_train[1]
    end = time.time()
    print('trainset created', end - start)

    start = time.time()
    images_test = images__.upload_images_test(round(n * 0.1))
    test_images, test_labels = images_test[0], images_test[1]
    end = time.time()
    print('testset created', end - start)

    start = time.time()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    end = time.time()
    print('images normalized', end - start)

    return [train_images, train_labels, test_images, test_labels]


def create_model():
    input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE * 2, 3))

    layer_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input_img)
    layer_1 = Conv2D(10, (3, 3), padding='same', activation='relu')(layer_1)

    layer_2 = Conv2D(10, (1, 1), padding='same', activation='relu')(input_img)
    layer_2 = Conv2D(10, (5, 5), padding='same', activation='relu')(layer_2)

    layer_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    layer_3 = Conv2D(10, (1, 1), padding='same', activation='relu')(layer_3)

    mid_1 = Concatenate(axis=3)([layer_1, layer_2, layer_3])

    flat_1 = Flatten()(mid_1)

    dense_1 = Dense(1200, activation='relu')(flat_1)
    dense_2 = Dense(600, activation='relu')(dense_1)
    dense_3 = Dense(150, activation='relu')(dense_2)
    output = Dense(2, activation='softmax')(dense_3)
    model = Model([input_img], output)

    print('layers added')

    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print('model compiled')
    return(model)

model_k = 0
his_train = []
his_test = []
def model_training(model_, train_images, train_labels, epochs):
    global model_k
    history_training = model_.fit(train_images, train_labels, epochs=epochs)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    # graph plot
    his_train.extend(history_training.history['accuracy'])
    temp = [test_acc]*epochs
    his_test.extend(temp)

    if model_k == 10:
        print(his_train)
        print(his_test)
        plt.plot(his_train, label='train accuracy')
        plt.plot(his_test, label='test accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.3, 1])
        plt.legend(loc='lower right')

        plt.show()

    model_k += 1
    return model_


percent_limit = 100
save_model_auto = 249
save_result_auto = 50
model_number_limit = 2
absolute_limit = 500

continue_ = True
same_number = 0
epochs = 2
epoch_size = 19000
new_images = True
n_testing = 1
epoch_index = 0
model = create_model()
automatically = False
save_folder = '0'
changing_image = 0
logfile = ''

while continue_:
    if not automatically and epoch_index != 0:
        print('do you want automatically? (0 = no, n = yes)')
        # input_auto = int(input())

        input_auto = 0


        if input_auto != 0:
            print('limit accuracy by percent (0 - 100)')
            percent_limit = int(input())
            print('save model every ... epochs (1 - n)')
            save_model_auto = int(input())
            print('how many epochs to save results? (1 - n)')
            save_result_auto = int(input())
            print('limit number of saved model (1 - 2n)')
            model_number_limit = int(input())
            print('absolute epoch limit (1 - n)')
            absolute_limit = int(input())
            print('does it correnct: (n = no, 1 yes')
            print('limit accuracy', percent_limit/100)
            print('save model every', save_model_auto, 'epochs')
            print('every', save_result_auto, 'epochs will save results on log files')
            print('limit number of saved model is', model_number_limit)
            print('absolute limit', absolute_limit)
            input_correct_auto = int(input())
            if input_correct_auto == 1:
                automatically = True

    #
    # automatization
    #
    if automatically:
        if epoch_index == 1:
            logfile += str(epoch_size) + ' images'
        if epochs * epoch_index >= absolute_limit:
            continue_ = False
            break

        # saving
        same_number = 2  # just in cause
        folder_0_list = os.listdir(base_dir + '0/')
        folder_1_list = os.listdir(base_dir + '1/')

        if len(folder_0_list) >= model_number_limit/2:
            save_folder = '1'
            if len(folder_1_list) >= model_number_limit/2:
                for model_ in folder_1_list:
                    os.remove(base_dir + '1/' + model_)

        if len(folder_1_list) >= model_number_limit:
            save_folder = '0'
            if len(folder_0_list) >= model_number_limit/2:
                for model_ in folder_0_list:
                    os.remove(base_dir + '0/' + model_)

        if epochs * epoch_index % save_model_auto == 0:
            model.save(base_dir + save_folder + '/saved_model-' + str(epochs*epoch_index) + '.h5')
            print('automatically saved')

        # updating images and contolling
        d_num = int(round(epoch_size/2))
        auto_loss, auto_acc = model.evaluate(train_images[:d_num], train_labels[:d_num])
        if auto_acc >= percent_limit/100:

            model.save(base_dir + '2/' + '/saved_model_super-' + str(epochs * epoch_index) + '.h5')

            new_images = True
            logfile += '\n'
            print('automatically updated')

        # writing log
        logfile += 'epoch:' + str(epochs * epoch_index) +'  trainset loss:' + str(auto_loss) \
                   + '  trainset accuracy:' + str(auto_acc) + '  used images:' + str(changing_image * epoch_size) \
                   + '  ||  testset accuracy:'+ str(test_acc) + '  testset loss:' + str(test_loss) + '\n'
        if epochs * epoch_index % save_result_auto == 0 and epochs * epoch_index != 0:
            print(logfile)
            print(type(logfile))
            f = open('D:/python/recognition/training_log.txt', 'a')
            f.write('\n\n' + str(logfile))
            f.close()
            logfile = ''
            print('automatically writed')


    #
    # inputing
    #
    if same_number <= 0:
        print('how long continue same? (0 = end, -n = new settings, n = same number)')
        print('setting now: epochs =', epochs, ' ,epoch_size = ', epoch_size, ' ,n_testing = ', n_testing)
        input_ = int(input())
        if input_ == 0:
            continue_ = False
            break
        if input_ < 0:
            print('epochs (0 = end, -n = same')
            print('now its', epochs)
            input_epochs = int(input())
            if input_epochs == 0:
                continue_ = False
                break
            if input_epochs > 0:
                epochs = input_epochs

            print('images (0 = same, n = change, -n = end )')
            print('now its', epoch_size)
            input_epoch_size = int(input())
            if input_epoch_size > 0:
                new_images = True
                epoch_size = input_epoch_size
            else:
                continue_ = False
                break

            print('number of testing images (0 = same, n = change, -n = end)')
            print('now its', n_testing)
            input_n_testing = int(input())
            if input_n_testing > 0:
                n_testing = input_n_testing
            else:
                continue_ = False
                break

            print('same namber (-n = end, n = same number')
            input_same_number = int(input())
            if input_same_number >= 0:
                same_number = input_same_number
            else:
                continue_ = False
                break
        else:
            same_number = input_
    else:
        same_number -= 1
        print('same:', same_number)

    if new_images:

        train_images, train_labels, test_images, test_labels = create_images(epoch_size)
        new_images = False
        changing_image += 1

    #
    # Training
    #
    model = model_training(model,train_images, train_labels, epochs)
    # model.fit(train_images, train_labels, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    predictions = model.predict(test_images)
    print('predicted')
    for i in range(n_testing):
        if len(predictions) < n_testing-1:
            print(np.argmax(predictions[i]), ' = ', test_labels[i])
    epoch_index += 1
    print(epochs * epoch_index)

print('save model? (0 = not save)')
if int(input()) != 0 and not automatically:
    model.save('comparison.h5')
    print('saved')