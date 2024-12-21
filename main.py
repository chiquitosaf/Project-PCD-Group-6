import streamlit as st
import cv2 as cv
import pickle
import numpy as np
from PIL import Image
import joblib

dict_nominal = {0 : "1000",
                1 : "10000",
                2 : "100000",
                3 : "2000",
                4 : "20000",
                5 : "5000",
                6 : "50000"}
k = 150
SIFT = cv.SIFT_create(nfeatures=200, enable_precise_upscale = True)
kmeans = joblib.load('kmeans_model2.joblib')
clf = joblib.load('rupiah_nominal_cls_model_82pt.joblib')

def prediction(image):
    # #turn image into grayscale
    img = Image.open(image)
    img = img.save("img.jpg")
    image = cv.imread("img.jpg")
    # image = cv.imread(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (224, 224), interpolation=cv.INTER_LINEAR)
    _, descriptors = SIFT.detectAndCompute(image, None)
    if descriptors is not None:
      words = kmeans.predict(descriptors)
      histogram, _ = np.histogram(words, bins=np.arange(k+1))
    prediction = clf.predict(histogram.reshape(1,-1))
    return dict_nominal[prediction[0]]

def test():
    return type(clf)

def main():
    st.title("Deteksi Nominal Uang Rupiah")
    st.write("Chiquito Shaduq Aurick Fulvian - 1301210284")
    st.write("Mochammad Rafi Adiyana - 1301210508")
    if 'image' not in st.session_state.keys():
        st.session_state['image'] = None
    cam_image = st.camera_input(label="Ambil Gambar", key="cam-image")
    if cam_image is not None:
        st.session_state['image'] = cam_image
    if st.session_state['image'] is not None:
        image = st.session_state['image']
        pred = prediction(image)
        # test1 = test()
        # st.image(image, caption=f"Nominal Uang: Rp {dict_nominal[prediction[0]]}", use_column_width=True)
        st.write(f"Nominal Rupiah : 1000")
    else:
        st.write("Silahkan ambil gambar terlebih dahulu")
    # image = '100k1.jpeg'
    # pred = prediction(image)
    # st.write(f"Nominal Rupiah : {pred}")

if __name__ == '__main__':
    main()
