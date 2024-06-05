import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf # Tensorflow version 2.13.0

st.header('Image Class predictor')

def main():
    file_upload = st.file_uploader('Choose the file', type=['jpg','png','jpeg'])
    if file_upload is not None:
        image = Image.open(file_upload)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = ''
        st.write(result)
        st.pyplot(figure)


def predict_class(images):
    classifier_model = tf.keras.models.load_model(r'd:\\Downloads\\model_efficentnetB3_90_Animal.h5')
    categori_classname = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle',
                      'bison', 'boar', 'butterfly', 'cat', 'caterpillar',
                      'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab',
                      'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly',
                      'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox',
                      'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper',
                      'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill',
                      'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo',
                      'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster',
                      'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan',
                      'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes',
                      'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat',
                      'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark',
                      'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish',
                      'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf',
                      'wombat', 'woodpecker', 'zebra']

    # Resize test image to match model input size (224x224)
    test_img_resized = images.resize((224, 224))
    # Convert test image to numpy array
    #test_input = np.array(test_img_resized)
    test_input = tf.keras.applications.efficientnet.preprocess_input(test_img_resized)
    # Expand dimensions to match model input shape (add batch dimension)
    test_input = np.expand_dims(test_input, axis=0)
    #test_input = test_input / 255.0
    y_pre = classifier_model.predict(test_input)
    #y_classes = [np.argmax(y_pre)][0]
    y_classes = np.argmax(y_pre)
    print(y_classes)
    print(categori_classname[y_classes])
    result = 'The image uploaded is: {}'.format(categori_classname[y_classes])
    return result


if __name__ == '__main__':
    main()