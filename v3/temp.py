import numpy as np
import librosa
import soundfile as sf
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import time

from scipy.io import wavfile
import noisereduce as nr

from time_domain_adaptive_filters.lms import lms
from time_domain_adaptive_filters.nlms import nlms
from time_domain_adaptive_filters.blms import blms
from time_domain_adaptive_filters.bnlms import bnlms
from time_domain_adaptive_filters.rls import rls
from time_domain_adaptive_filters.apa import apa
from time_domain_adaptive_filters.kalman import kalman
from frequency_domain_adaptive_filters.pfdaf import pfdaf
from frequency_domain_adaptive_filters.fdaf import fdaf
from frequency_domain_adaptive_filters.fdkf import fdkf
from frequency_domain_adaptive_filters.pfdkf import pfdkf
from nonlinear_adaptive_filters.volterra import svf
from nonlinear_adaptive_filters.flaf import flaf
from nonlinear_adaptive_filters.aeflaf import aeflaf
from nonlinear_adaptive_filters.sflaf import sflaf
from nonlinear_adaptive_filters.cflaf import cflaf

near='/home/mher/Project_Python/v3/samples/nearspeech.wav'
far='/home/mher/Project_Python/v3/samples/farspeech.wav'
mic='/home/mher/Project_Python/v3/samples/micspeechecho.wav'

def genEcho(nearpath, farpath, micpath):
    #Load near-end speech
    nearspeech, fs = sf.read(nearpath)
    #Load far-end speech
    farspeech, fs = sf.read(farpath)

    #plt.figure()
    #plt.subplot(211)
    #plt.plot(farspeech)
    #plt.subplot(212)
    #plt.plot(nearspeech)

    #Room Impulse Response
    # room dimension
    room_dim = [5, 4, 6]
    # Create the shoebox
    room = pra.ShoeBox(
        room_dim,
        absorption=0.0,
        fs=fs,
        max_order=15,
        )
    # source and mic locations
    room.add_source([2, 2.1, 2], signal=farspeech)
    #room.add_source([2, 1.9, 2], signal=nearspeech)
    
    room.add_microphone_array(
            pra.MicrophoneArray(
                np.array([[2, 2, 2]]).T, 
                room.fs)
            )

    # run ism
    room.simulate()

    #room.mic_array.to_wav(micpath, norm=True, bitdepth=np.int16)
    farspeechecho = room.mic_array.signals[0,:]
    return farspeechecho
    #return room.rir[0][0]

#fsecho = genEcho(near, far, mic)
#plt.plot(fsecho)
#plt.show()
#plt.subplot(211)
#plt.plot(fsecho)
#plt.subplot(212)
d, sr  = librosa.load('/home/mher/Project_Python/v3/samples/micspeech.wav',sr=8000)
#plt.plot(d)
#plt.show()
x, sr  = librosa.load('/home/mher/Project_Python/v3/samples/farspeech.wav',sr=8000)

#sf.write('/home/mher/Project_Python/v3/samples/echo.wav', fsecho, sr, subtype='PCM_16')
e= kalman(x, d, N=64)
e = np.clip(e,-1,1)
sf.write('/home/mher/Project_Python/v3/samples/result.wav', e, sr, subtype='PCM_16')

# load data
rate, data = wavfile.read("/home/mher/Project_Python/v3/samples/result.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("/home/mher/Project_Python/v3/samples/result_without_noise.wav", rate, reduced_noise)
