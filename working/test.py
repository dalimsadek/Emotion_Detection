import matplotlib.pyplot as plt
import train


if __name__=="__main__":

    plt.style.use('dark_background')

    plt.figure(figsize=(20,10))
    plt.subplot(1, 2, 1)
    plt.suptitle('Optimizer : Adam', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(train.history.history['loss'], label='Training Loss')
    plt.plot(train.history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(train.history.history['accuracy'], label='Training Accuracy')
    plt.plot(train.history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.show()