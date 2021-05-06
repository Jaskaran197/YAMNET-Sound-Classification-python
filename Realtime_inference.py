#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf

import params
import yamnet as yamnet_model

import pyaudio
from array import array
import wave


# In[2]:


interpreter = tf.lite.Interpreter(model_path="yamnet.tflite")
interpreter.allocate_tensors()
inputs = interpreter.get_input_details()
outputs = interpreter.get_output_details()

yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')




def classify_sound(file_name):
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
      waveform = np.mean(waveform, axis=1)
    if sr != params.SAMPLE_RATE:
      waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)

    interpreter.set_tensor(inputs[0]['index'], np.expand_dims(np.array(waveform, dtype=np.float32), axis=0))
    interpreter.invoke()
    scores = interpreter.get_tensor(outputs[0]['index'])

    # Scores is a matrix of (time_frames, num_classes) classifier scores.
    # Average them along time to get an overall classifier output for the clip.
    prediction = np.mean(scores, axis=0)
    # Report the highest-scoring classes and their scores.
    top5_i = np.argsort(prediction)[::-1][:5]
    print( '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                    for i in top5_i))


# In[32]:


#classify_sound(file_name)


# In[33]:





# In[34]:


FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=16000
CHUNK=50
CHUNK_THRESH=64
RECORD_SECONDS=0.975
FILE_NAME="RECORDING.wav"
while(True):
    audio=pyaudio.PyAudio() #instantiate the pyaudio

    #recording prerequisites
    stream=audio.open(format=FORMAT,channels=CHANNELS, 
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK)

    #starting recording
    frames=[]
    while(True):
        old_data=stream.read(CHUNK_THRESH)
        data_chunk=array('h',old_data)
        vol=max(data_chunk)
        #print(old_data)
        if(vol>=200):
            print('Triggered')
            #frames.append(old_data)
            break
    for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
        #frames.append(old_data)
        data=stream.read(CHUNK)
        data_chunk=array('h',data)
        vol=max(data_chunk)
        #if(vol>=300):
         #   print("something said")
        frames.append(data)
        #else:
            #print("nothing")
        #print("\n")


    #end of recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    #writing to file
    wavfile=wave.open(FILE_NAME,'wb')
    wavfile.setnchannels(CHANNELS)
    wavfile.setsampwidth(audio.get_sample_size(FORMAT))
    wavfile.setframerate(RATE)
    wavfile.writeframes(b''.join(frames))#append frames recorded to file
    wavfile.close()
    
    classify_sound(FILE_NAME)
       


# In[ ]:




