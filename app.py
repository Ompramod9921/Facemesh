import streamlit as st
import mediapipe as mp
from PIL import Image,ImageDraw
import numpy as np
import cv2


st.set_page_config(page_title='Facemesh',page_icon='👽')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Face Landmark Detection")
st.write("Made with ❤️ by om pramod")
st.markdown("*****")

image_file = st.file_uploader("upload your selfie",type=["png","jpg","jpeg"])

if image_file is not None :
    try :
        st.image(image_file,use_column_width=True)
        image_loaded = Image.open(image_file)
        new_image = np.array(image_loaded.convert('RGB')) #converting image into array
        img = cv2.cvtColor(new_image,1) #converting the image from 3 channel image (RGB) into 1 channel image.if you don't convert the image into one channel, open-cv does it automatically.

        # Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

        # Facial landmarks
        result = face_mesh.process(img)

        height, width, _ = img.shape

        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                new = cv2.circle(img, (x, y), 2, (0,0,255), -1)
        if st.button("Draw Facemesh"):
            st.markdown("****")
            st.image(new,use_column_width=True)
    except :
        st.error("Face could not be detected. Please confirm that the picture is a face photo")      

   
