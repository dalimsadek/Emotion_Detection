from dependencies import *




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

    epochs = 48

    model.compile(loss='categorical_crossentropy',
                optimizer = Adam(lr=0.001),
                metrics=['accuracy'])


    history = model.fit_generator(generator=train_set,
                                    steps_per_epoch=train_set.n//train_set.batch_size,
                                    epochs=epochs,
                                    validation_data = test_set,
                                    validation_steps = test_set.n//test_set.batch_size,
                                    callbacks=callbacks_list
                                    )