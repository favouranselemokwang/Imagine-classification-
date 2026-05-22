# model.py - Configuration: Custom_Sequential_Deep for Client 30
import tensorflow as tf

def build_model(num_classes=3):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    base_model = model
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0011),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model
