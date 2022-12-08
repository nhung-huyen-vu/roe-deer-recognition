from roedeer_gender_classifier import RoeDeerGenderClassificationModel
from dataset import RoeDeerDataGen
import tensorflow as tf
import sys

def train(data_path):
    batch_size = 8 
    initial_epochs = 10
    tuning_epochs = 20
    finetune_from_layer = 160

    model, base_resnet, base_month, head = RoeDeerGenderClassificationModel()
   
    # Freeze resnet base
    for layer in base_resnet.layers:
        layer.trainable = False

    # Create generators for train and validation data with batch size and 4 classes
    train_generator = RoeDeerDataGen(data_path, "train", batch_size, 2, balance_classes = 
            True)
    validation_generator = RoeDeerDataGen(data_path, "valid", batch_size, 2, balance_classes = False)

    # Create callback for checkpoint creation
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="best_pretrain.model",
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=opt, metrics=['accuracy'])

    # Fit model for initial epochs
    history = model.fit(x=train_generator, validation_data=validation_generator, batch_size=batch_size, epochs=initial_epochs, verbose=1, callbacks=[model_checkpoint_callback])

    model = tf.keras.models.load_model("best_pretrain.model")
    
    # Create callback for early stopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)

    # Create callback for checkpoint creation
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="best.model",
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Unfreeze part of base
    for layer in base_resnet.layers:
        layer.trainable = True

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=opt, metrics=['accuracy'])

    history_fine = model.fit(x=train_generator, validation_data=validation_generator, batch_size=batch_size, initial_epoch=history.epoch[-1], epochs=initial_epochs + tuning_epochs, verbose=1, callbacks=[model_checkpoint_callback])

    model.save("final.model")


data_path = sys.argv[1]
train(data_path)






