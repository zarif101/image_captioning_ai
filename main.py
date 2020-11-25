import streamlit as st
import numpy as np
import pandas as pd
from utils import caption_path,caption_img
import cv2
from PIL import Image

def main():
    mode = st.sidebar.selectbox(
    "Select mode",
    ("Example", "Try it!",)
)
    st.title('Welcome to Capper! 😎')
    st.header('AI Powered Automated Image Captioning')
    st.write('Note: I cannot ensure the accuracy of the captions, and this is very much a work in progress')
    if mode == 'Example':

        st.header('Example caption')
        example_img = cv2.imread('static/dog.jpg')
        example_img = cv2.cvtColor(example_img,cv2.COLOR_BGR2RGB)
        example_img = cv2.resize(example_img,(224,224))
        st.image(example_img)
        st.header(caption_path('static/dog.jpg'))
    else:
        uploaded_image = st.file_uploader("Upload an image!",type=['png','jpg','jpeg'])
        st.write('If it is taking a long time, then your image may be too diffcult - please reload the page and try again with another picture..sorry')
        if uploaded_image:
            pic = Image.open(uploaded_image)
            pic = np.array(pic)
            pic = cv2.resize(pic,(224,224))
            st.image(pic)
            st.header(caption_img(pic))
            st.write('To try another picture, press the x and select a different photo!')
if __name__=='__main__':
    main()
