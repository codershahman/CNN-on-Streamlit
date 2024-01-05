import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from util import classify, set_background


#Setting the title
st.title("Fashion Classifier using the MNIST Dataset")

#Including a header 
st.header("Upload the image of your clothing item!")
st.markdown("Note: It would be best if you go to my Github and download some images from the 'Images to Test' folder :)")


#Getting the user input
file=st.file_uploader("", type=('jpeg','jpg','png'))


model=load_model('weights.hdf5')

with open('labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
