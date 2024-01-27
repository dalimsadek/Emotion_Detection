from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import split_data as sp
import model 
from keras.optimizers import Adam,SGD,RMSprop
import config



if __name__=="__main__":

    checkpoint = ModelCheckpoint("./model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    early_stopping = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=3,
                            verbose=1,
                            restore_best_weights=True
                            )

    reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                                factor=0.2,
                                patience=3,
                                verbose=1,
                                min_delta=0.0001)

    callbacks_list = [early_stopping,checkpoint,reduce_learningrate]


    model.model.compile(loss='categorical_crossentropy',
                optimizer = Adam(lr=config.learning_rate),
                metrics=['accuracy'])


    history = model.model.fit_generator(generator=sp.train_set,
                                    steps_per_epoch=sp.train_set.n//sp.train_set.batch_size,
                                    epochs=config.epochs,
                                    validation_data = sp.test_set,
                                    validation_steps = sp.test_set.n//sp.test_set.batch_size,
                                    callbacks=callbacks_list
                                    )