from dependencies import *

if __name__=="__main__":

    picture_size = 48
    folder_path = "../input/face-expression-recognition-dataset/images/"



    batch_size  = 128

    datagen_train  = ImageDataGenerator()
    datagen_val = ImageDataGenerator()



    train_set = datagen_train.flow_from_directory(folder_path+"train",
                                              target_size = (picture_size,picture_size),
                                              color_mode = "grayscale",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=True)


    test_set = datagen_val.flow_from_directory(folder_path+"validation",
                                              target_size = (picture_size,picture_size),
                                              color_mode = "grayscale",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=False)