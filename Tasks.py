#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import cv2
import io
import time
from PIL import Image, ImageOps


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

result = st.button("Task 6")
if result:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
    cap.set(cv2.CAP_PROP_FPS, 30)

    x, y, w, h = 800, 500, 100, 100
    x, y, w, h = 950, 300, 100, 100
    heartbeat_count = 128
    heartbeat_values = [0]*heartbeat_count
    heartbeat_times = [time.time()]*heartbeat_count
    fig = plt.figure()
    ax = fig.add_subplot(111)
    while(True):
    
        ret, frame = cap.read()    
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop_img = img[y:y + h, x:x + w]    
        heartbeat_values = heartbeat_values[1:] + [np.average(crop_img)]
        heartbeat_times = heartbeat_times[1:] + [time.time()]    
        ax.plot(heartbeat_times, heartbeat_values)
        fig.canvas.draw()
        plot_img_np = np.fromstring(fig.canvas.tostring_rgb(),
                                dtype=np.uint8, sep='')
        plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.cla()    
        cv2.imshow('Crop', crop_img)
        cv2.imshow('Graph', plot_img_np)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
cv2.destroyAllWindows()

# In[ ]:




