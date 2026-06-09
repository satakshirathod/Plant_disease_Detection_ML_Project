import tensorflow as tf
model = tf.keras.models.load_model('potato_disease_model.h5', compile=False)
for layer in model.layers:
    if 'base_model' in layer.name or 'mobilenetv2' in layer.name:
        for sub_layer in layer.layers:
             print(sub_layer.name)