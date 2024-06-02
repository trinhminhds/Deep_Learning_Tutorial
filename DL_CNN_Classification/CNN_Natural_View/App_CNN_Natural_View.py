import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

st.header('Image class predictor')

def main():
    file_upload = st.file_uploader('Choose the file',type = ['jpg','png','jpeg'])
    if file_upload is not None:
        image = Image.open(file_upload)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)


def predict_class(images):
    classifier_model = tf.keras.models.load_model(r'd:\\Downloads\\Resnet50v2_Building_Sea.h5')
    cate = ['Building','Forest','Glacier','Mountain','Sea','Street']
    test_img_resized = images.resize((244, 244))
    test_input = np.array(test_img_resized)
    test_input = np.expand_dims(test_input, axis=0)
    test_input = test_input / 255.0
    y_pre = classifier_model.predict(test_input)
    y_classes = [np.argmax(y_pre)][0]
    print(y_pre)
    print(cate[y_classes])
    result = 'The image uploaded is: {}'.format(cate[y_classes])
    return result
    

if __name__ == '__main__':
    main()
    
