from roedeer_gender_classifier import RoeDeerGenderClassificationModel
from dataset import RoeDeerDataGen
import tensorflow as tf
import sys

def train(data_path):
    batch_size = 32

    model, base_resnet, base_month, head = RoeDeerGenderClassificationModel()
   
    # Freeze resnet base
    for layer in base_resnet.layers:
        layer.trainable = False

    # Create callback for early stopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)

    # Create callback for checkpoint creation
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints",
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Create generators for train and validation data with batch size and 4 classes
    train_generator = RoeDeerDataGen(data_path, "train", batch_size, 4, balance_classes = True)
    validation_generator = RoeDeerDataGen(data_path, "valid", batch_size, 4)

    # Compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Fit model
    model.fit(x=train_generator, validation_data=validation_generator, batch_size=batch_size, epochs=200, verbose=1, callbacks=[early_stopping_callback, model_checkpoint_callback])

data_path = sys.argv[1]
train(data_path)






