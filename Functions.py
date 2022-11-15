import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import math
import streamlit as st
import  streamlit_vertical_slider  as svs
import io
from io import BytesIO 
from scipy.io.wavfile import write
from scipy import signal
import altair as alt
import time
import pandas as pd
import altair as alt
import plotly.graph_objects as go




class Functions():
    def  mode_1_ranges(sample_rate):
        fMax = sample_rate//2
        list_of_range = np.arange(0,fMax,fMax/11)
        list_of_lists =[]
        for i in range(len(list_of_range)-1):
            small_list=[list_of_range[i],list_of_range[i+1]]
            list_of_lists.append(small_list)
        return list_of_lists
    def final_func(fou_of_signal,frequencies,list_of_freqs,list_of_sliders):
        final_fou = fou_of_signal
        for iter in range(len(list_of_sliders)):
            freqs_update = Functions.select_range(frequencies,list_of_freqs[iter][0],list_of_freqs[iter][1],True)
            final_fou = Functions.modify_magnitude(freqs_update,fou_of_signal,list_of_sliders[iter])
        return final_fou
    def read_ecg_file(file_path):
        signal = loadmat(file_path)
        signal = signal['val'][0]
        return signal
    def modify_magnitude(freq_list,list_freq_domain,factor):
        list_freq_domain[freq_list] = list_freq_domain[freq_list] * (factor+1)
        return list_freq_domain
    def fourier_transformation(time_domain_data, sampling_rate):
        frequncies = np.fft.rfftfreq(len(time_domain_data), 1/sampling_rate)
        freq_domain_data = np.fft.rfft(time_domain_data)
        phase = np.angle(freq_domain_data)
        magnitude = np.abs(freq_domain_data)
        return freq_domain_data, frequncies, magnitude, phase, len(time_domain_data)
    def select_range(frequncies,min,max,equal):
        if equal:
            selected_freqs = (frequncies>=min)&(frequncies<=max)
        else:
            selected_freqs = (frequncies>min)&(frequncies<max)
        return 

    def bins_separation(frequency, amplitude,sliders_number):
        freq_list = []
        amplitude_list = []
        bin_max_frequency_value = math.ceil(len(frequency)/sliders_number)
        i = 0
        for i in range(0, sliders_number):
            freq_list.append(frequency[i*bin_max_frequency_value : (i+1)*bin_max_frequency_value])
            amplitude_list.append(amplitude[i*bin_max_frequency_value:(i+1)*bin_max_frequency_value])

        return freq_list, amplitude_list,bin_max_frequency_value

    def Sliders_generation( sliders_number):
            columns = st.columns(sliders_number)
            values = []
            for i in range(0, sliders_number):
                with columns[i]:
                    value = svs.vertical_slider( key= i, default_value=0.0, step=0.1, min_value=-1.0, max_value=1.0)
                    if value == None:
                        value = 0.0
                    values.append(value)
                    
            return values   

        
    def convertToAudio(sr,signal):
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io,sr, signal.astype(np.float32))
        result_bytes = byte_io.read()
        return result_bytes

    def inverse(amp,phase):
        combined=np.multiply(amp,np.exp(1j*phase))
        inverse_combined=np.fft.irfft(combined)
        signal=np.real(inverse_combined)
        return signal

    def plotSpectrogram(audioData, fs, Title):
        N = 512
        w = signal.blackman(N)
        freqs, time, Pxx = signal.spectrogram(audioData, fs, window=w, nfft=N)

        layout = go.Layout(margin=go.layout.Margin(l=0, r=0, b=0, t=30,))
        fig = go.Figure(layout=layout)

        fig.add_trace(go.Heatmap(x=time, y=freqs, z=10*np.log10(Pxx),
                    colorscale='Jet', name='Spectrogram'))

        fig.update_layout(height=300, title={
            'text': Title,
            'y': 1,
            'x': 0.49,
            'xanchor': 'center',
            'yanchor': 'top'},
            title_font=dict(
            family="Arial",
            size=17))
        fig.update_xaxes(title='Time')
        fig.update_yaxes(title='Frequency')
        st.sidebar.plotly_chart(fig, use_container_width=True)



    # def altair_plot(original_df, modified_df):
    #     lines = alt.Chart(original_df).mark_line().encode(
    #         x=alt.X('0:T', axis=alt.Axis(title='Time')),
    #         y=alt.Y('1:Q', axis=alt.Axis(title='Amplitude'))
    #     ).properties(
    #         width=400,
    #         height=300
    #     )
    #     modified_lines = alt.Chart(modified_df).mark_line().encode(
    #         x=alt.X('0:T', axis=alt.Axis(title='Time')),
    #         y=alt.Y('1:Q', axis=alt.Axis(title='Amplitude'))
    #     ).properties(
    #         width=400,
    #         height=300
    #     ).interactive()
    #     return lines


    # def animation(original_df):
    #     brush = alt.selection_interval()
    #     lines = alt.Chart(original_df).mark_line().encode(
    #         x=alt.X('time', axis=alt.Axis(title='Time')),
    #         y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude')),
    #     ).properties(
    #         width=400,
    #         height=300
    #     ).add_selection(
    #         brush).interactive()

    #     return lines


    # def dynamic_plot(line_plot, original_df, modified_df):
    #     N = len(original_df)
    #     j=0
    #     size = 10 
    #     for i in range(0, N):
    #         step_df = original_df.iloc[i:(i+1)*round((0.05*N))]
    #         mod_step_df = modified_df.iloc[i:(i+1)*round((0.05*N))]
    #         lines = Functions.animation(step_df)
    #         mod_lines = Functions.animation(mod_step_df)
    #         concat = alt.hconcat(lines, mod_lines)
    #         line_plot = line_plot.altair_chart(concat)
    #         time.sleep(.15)
    #         j+=round((0.05*N))
    #         if j>N:
    #             break

    # def plotFunc(t,amplitude,amplitude_inv,start_btn,resume_btn,pause_btn):
    #     original_df = pd.DataFrame(
    #                 {'time': t, 'amplitude': amplitude}, columns=['time', 'amplitude'])
    #     modified_df = pd.DataFrame(
    #                 {'time': t, 'amplitude': amplitude_inv}, columns=['time', 'amplitude'])
    #     lines = Functions.altair_plot(original_df, modified_df)
            
    #     line_plot = st.altair_chart(lines)
    #     line_plot= line_plot.altair_chart(lines)

    #     if start_btn:
    #         st.session_state.flag = 1
    #         Functions.dynamic_plot(line_plot, original_df, modified_df)
    #     elif resume_btn:
    #         st.session_state.flag = 1
    #         Functions.dynamic_plot(line_plot, original_df, modified_df)


    #     elif pause_btn:
    #         st.session_state.flag =0
    #         Functions.dynamic_plot(line_plot, original_df, modified_df)


    #     if st.session_state.flag == 1:
    #         Functions.dynamic_plot(line_plot, original_df, modified_df)
        # Functions.dynamic_plot(line_plot, original_df, modified_df)

    def slider_group(groups): 
        adjusted_data = []
        sliders = {}
        columns = st.columns(len(groups))
        for idx, i in enumerate(groups):
            min_value = i[0]
            max_value = i[1]
            key = f'member{str(idx)}'
            with columns[idx]:
                sliders[f'slider_group_{key}'] = svs.vertical_slider(key=key, default_value=i[2], step=1, min_value=min_value, max_value=max_value)
                st.text(i[4])
                if sliders[f'slider_group_{key}'] == None:
                    sliders[f'slider_group_{key}']  = i[2]
                adjusted_data.append((i[3],(sliders[f'slider_group_{key}'] )))
        return adjusted_data


    def plot_animation(df):
        brush = alt.selection_interval()
        chart1 = alt.Chart(df).mark_line().encode(
                x=alt.X('time', axis=alt.Axis(title='Time')),
                # y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude')),
            ).properties(
                width=500,
                height=300
            ).add_selection(
                brush).interactive()
        
        figure = chart1.encode(
                    y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude')))| chart1.encode(
                    y=alt.Y('amplitude after processing',axis=alt.Axis(title='Amplitude after'))).add_selection(
                brush)

        return figure

    def plotShow(data, idata,start_btn,pause_btn,resume_btn,value,sr):

        time1 = len(data)/(sr)
        if time1>1:
            time1 = int(time1)
        time1 = np.linspace(0,time1,len(data))   
        df = pd.DataFrame({'time': time1[::300], 
                            'amplitude': data[:: 300],
                            'amplitude after processing': idata[::300]}, columns=[
                            'time', 'amplitude','amplitude after processing'])
        N = df.shape[0]  # number of elements in the dataframe
        burst = 10      # number of elements (months) to add to the plot
        size = burst 
        
        step_df = df.iloc[0:st.session_state.size1]
        if st.session_state.size1 ==0:
            step_df = df.iloc[0:N]

        lines = Functions.plot_animation(step_df)
        line_plot = st.altair_chart(lines)
        line_plot= line_plot.altair_chart(lines)

        # lines = plot_animation(df)
        # line_plot = st.altair_chart(lines)
        N = df.shape[0]  # number of elements in the dataframe
        burst = 10      # number of elements (months) to add to the plot
        size = burst    #   size of the current dataset
        if start_btn:
            st.session_state.flag = 1
            for i in range(1, N):
                st.session_state.start=i
                step_df = df.iloc[0:size]
                lines = Functions.plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                size = i + burst 
                st.session_state.size1 = size
                time.sleep(.1)

        elif resume_btn: 
                st.session_state.flag = 1
                for i in range( st.session_state.start,N):
                    st.session_state.start =i 
                    step_df = df.iloc[0:size]
                    lines = Functions.plot_animation(step_df)
                    line_plot = line_plot.altair_chart(lines)
                    st.session_state.size1 = size
                    size = i + burst
                    time.sleep(.1)

        elif pause_btn:
                st.session_state.flag =0
                step_df = df.iloc[0:st.session_state.size1]
                lines = Functions.plot_animation(step_df)
                line_plot= line_plot.altair_chart(lines)



        if st.session_state.flag == 1:
            for i in range( st.session_state.start,N):
                    st.session_state.start =i 
                    step_df = df.iloc[0:size]
                    lines = Functions.plot_animation(step_df)
                    line_plot = line_plot.altair_chart(lines)
                    st.session_state.size1 = size
                    size = i + burst
                    time.sleep(.1)