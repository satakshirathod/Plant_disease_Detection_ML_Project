import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

@tf.keras.utils.register_keras_serializable(package="Custom")
def custom_preprocess(x):
    return preprocess_input(x)

app = Flask(__name__)

try:
    model = tf.keras.models.load_model(
        'potato_disease_model.h5', 
        custom_objects={'preprocess_input': custom_preprocess},
        compile=False
    )
    print("✅ Model Active")
except Exception as e:
    print(f"❌ Error: {e}")

class_names = ['Early_Blight', 'Healthy', 'Late_Blight']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['imagefile']
        if file:
            filename = file.filename
            filepath = os.path.join('static', filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            # Note: Depending on your model training, you might need to call preprocess_input(x) here
            preds = model.predict(x)
            prediction = class_names[np.argmax(preds[0])]
            confidence = round(100 * np.max(preds[0]), 2)

            return render_template('index.html', 
                                   prediction=prediction, 
                                   confidence=confidence, 
                                   image_path=filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)