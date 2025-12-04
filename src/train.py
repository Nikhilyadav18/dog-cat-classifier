import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def build_model():
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False
    
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    
    val_gen = ImageDataGenerator(rescale=1./255)
    
    train_data = train_gen.flow_from_directory(
        "../data/train", # change this path accordingly 
        target_size=(224,224),
        batch_size=32,
        class_mode='binary'
    )
    
    val_data = val_gen.flow_from_directory(
        "../data/val", # change this path accordingly 
        target_size=(224,224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    
    model = build_model()
    
    checkpoint = ModelCheckpoint(".../models/mobilenetv2_best.h5", save_best_only=True, monitor="val_accuracy")
    early = EarlyStopping(patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3)
    
    model.fit(train_data, validation_data=val_data, epochs=20, callbacks=[checkpoint, early, reduce_lr])

if __name__ == "__main__":
    train()
