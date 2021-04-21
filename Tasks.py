#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np

# In[10]:
face = cv2.CascadeClassifier('C:/Users/kumar/Documents/intern/harsh.xml')

st.title("MetFlux Internship Tasks ")
st.write("Upload the Image:")
file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("Please upload an image")
else:
    image = Image.open(file)
    img = np.asarray(image)
    st.image(image, use_column_width = True)
    frame = img
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    gray_blur = cv2.GaussianBlur(gray, (25, 25), 0)

    
    cartoon = cv2.divide(gray, gray_blur, scale=250.0)

    st.image(frame, use_coloumn_width = True)
    st.image(cartoon, use_coloumn_width = True)
result = st.button("Tasks")
if result:


    cp = cv2.VideoCapture(0)
    while True:
        _, img = cp.read()
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, 1.1, 4)
    
        for(x, y, z, h) in faces:
            cv2.rectangle(img, (x, y), (x+z, y+h), (255, 0, 0), 2)
        
        cv2.imshow('img', img)
    
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cp.release()

# In[ ]:




