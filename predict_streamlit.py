# -*- encoding: utf-8 -*-
'''
@File    :   predict_streamlit.py
@Time    :   2022/03/25 16:33:23
@Author  :   MHIRI 
'''

# here put the import lib

import os
import streamlit as st 
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import keras
from numpy import argmax
import pandas as pd
import librosa






def footer_markdown():
    footer="""
    <style>
    a:link , a:visited{
    color: blue;
    background-color: transparent;
    text-decoration: underline;
    }
    
    a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
    }
    
    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    }
    </style>
    <div class="footer">
    <p>Developed by <a style='display: block; text-align: center;' >Rahma MHIRI</a></p>
    </div>
    """
    return footer

def save_file(file_path):
    # save your sound file in the right folder by following the path
    for wav_file in file_path:
    	with open(os.path.join(r'./audio_files', wav_file.name),'wb') as f:
         	f.write(wav_file.getbuffer())
    	


def wav2mfcc(file_path, max_pad_len=1251):

    
    
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=sr)
    
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def get_data(file_path,classes):

    labels = []
    mfccs = []
    file_list=[]
    
    
    #for wav_file in file_path:
    for wav_file in os.listdir(r'./audio_files'):
        #sound_name = f'audio_files/{sound_saved}'
        #sound = wav_file.getbuffer()
        mfccs.append(wav2mfcc(r'./audio_files/'+wav_file))
        n=wav_file
        file_list.append(n)

    mfccs=np.asarray(mfccs)
    file_wav=file_list

    dim_1 = mfccs.shape[1]
    dim_2 = mfccs.shape[2]
    channels = 1
    classes = classes

    X = mfccs
    X = X.reshape((mfccs.shape[0], dim_1, dim_2, channels))
    X_test = X

    return X_test, file_wav

def app():
    """
    Main function that contains the application for getting predictions from 
    keras based trained models.
    """
    
    # Get list of saved h5 models, which will be displayed in option to load.
    h5_file_list = [file for file in os.listdir("./model") if file.endswith(".h5")]
    h5_file_names = [os.path.splitext(file)[0] for file in h5_file_list]
    
    st.title("Batsort Prediction Basic UI")
    st.header("A Streamlit app based Web UI To Get Predictions From Trained Models")
    st.markdown(footer_markdown(),unsafe_allow_html=True)
    model_type = st.radio("Choose trained model to load...", h5_file_names)
    
    loaded_model = tf.keras.models.load_model("./model/{}.h5".format(model_type))
    
    uploaded_file = st.file_uploader("Choose a wav...", type="wav",accept_multiple_files = True)
    
    
    if uploaded_file is not None and uploaded_file :
        if "Virer_parasites" in model_type:
            classes = 2
        
        
        filelist = [ f for f in os.listdir(r'./audio_files') if f.endswith(".wav") ]
        for f in filelist:
        	os.remove(os.path.join(r'./audio_files', f))
        st.write("")
        st.write("Identifying...")
        save_file(uploaded_file)
        X, file_wav = get_data(uploaded_file, classes)
        
        luna=loaded_model.predict(X)
        indice = tf.argmax(luna,axis=1)
        indice_numpy =indice.numpy()
        luna_numpy_list=luna.tolist()
        
        
        predictions = loaded_model.predict_classes(X)
        pred=to_categorical(predictions)
        predicted_categories = tf.argmax(pred,axis=1)
        np_pred = predicted_categories.numpy()
        np_pred_list=np_pred.tolist()
        
    
        ID_true_list=[]
        ID_pred_list=[]
        wav_list=[]
        pred_class=[]
        for i in range(classes):
            for j in range(len(np_pred_list)):
                if np_pred_list[j]==i : 
                    
                    wav_list.append(file_wav[j])
                    pred_class.append(np_pred_list[j])
                    ID = luna_numpy_list[j]
                    ID_pred=ID[np_pred_list[j]]
                    ID_pred_list.append(ID_pred)
        
        for i in range(len(pred_class)):
            if pred_class[i] == 0:
                pred_class[i] = 'Parasi'
            if pred_class[i] == 1:
                pred_class[i] = 'Chiro'
            

        #st.write('%s' % (pred_class) )
        dictionary = {'Nom du son': wav_list, 'pred_class':pred_class, 'ID_pred':ID_pred_list}  
        dataframe = pd.DataFrame(dictionary) 
        result=dataframe.to_csv(index=False,header=True)
        href = f'Download csv file:'
        st.write(href, unsafe_allow_html=True)
        st.download_button(label="Download data as CSV",data=result,file_name='untitled.csv',mime='text/csv')
        st.write("NB:Please refresh the page for each loading for files")
       
            

                

if __name__=='__main__':
    app()
    
