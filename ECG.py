import streamlit as st
import matplotlib.pylab as plt 
import numpy as np
from numpy import fft
from scipy.io import loadmat
import plotly.express as px

uploaded_file = st.file_uploader('upload', label_visibility="hidden")
   
if uploaded_file is not None:
   Signal=loadmat(uploaded_file)
   Signal_value=Signal["val"][0]
   N = len(  Signal_value)
   st.write(   N)
   Fs = 1000
   T = 1 / Fs
   x=np.linspace(0,N*T,N)
   fig = px.line(x=x, y=Signal_value)
   st.plotly_chart(fig,use_container_width=True)

#Compute FFT
   yf = fft.fft(Signal_value)
   yf=np.abs(yf)
#Compute frequency x-axis
#    xf = np.linspace(0, 1/(2*T),N)
   xf=np.abs(fft.fftfreq(N,T))
   fig_2= px.line(x=xf, y=yf)
   st.plotly_chart(fig_2,use_container_width=True)
  
