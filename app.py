import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('model/Weather-Model.h5')
    return model


def import_and_predict(image_data, model):
    resized_image = tf.image.resize(image_data, (256,256))
    prediction = model.predict(np.expand_dims(resized_image/255, 0))
    return prediction

def runapp():
    st.write(""" ## Parag's Weather Classification App""")
    model = load_model()
    file = st.file_uploader("Upload image here", type=["jpg","png","jpeg"])
    
    if file is None:
        st.text("Please upload an Image file")
    else:
        img = Image.open(file)
        st.image(img, use_column_width=True)
        img = np.array(img)
        prediction = import_and_predict(img, model)
        class_names = ['Cloudy', 'Rainy', 'Shine','Sunrise']
        string = "Predicted class is " + class_names[np.argmax(prediction)]
        st.success(string)
        
if __name__ == "__main__":
    runapp()



