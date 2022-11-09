import streamlit as st
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

file_uploaded = st.file_uploader("")
slider_value = st.slider("phase" , min_value= -10 , max_value= 10)

if file_uploaded is not None:
                path=file_uploaded.name   # get path 
                time_series , samling_rate = librosa.load(path)
                st.audio(path)
                
                col1, col2 = st.columns(2)
                time = np.array(range(0,len(time_series)))/ samling_rate
                fig = px.line(x = time , y =time_series )
                
                with col1:
                    st.plotly_chart(fig , use_container_width=True)
                result = librosa.effects.pitch_shift(time_series , sr= samling_rate , n_steps= slider_value)
                sf.write("output_signal.wav" , result ,samling_rate)
                st.audio("output_singnal.wav")
                fig = px.line(x = time , y =result )
                
                with col2:
                    st.plotly_chart(fig , use_container_width=True)

