import numpy as np
import matplotlib.pylab as plt
import padasip as pa
from scipy.io import wavfile


errorsignal='/home/mher/Project_Python/Sounds/error.wav'
sfs, u = wavfile.read('/home/mher/Project_Python/Sounds/Sender.wav')
lfs, v = wavfile.read('/home/mher/Project_Python/Sounds/Listener.wav')
ofs, d = wavfile.read('/home/mher/Project_Python/Sounds/output.wav')
print(type(u))
print(type(d))
u = np.frombuffer(u, np.int16)
u = np.float64(v)

v = np.frombuffer(v, np.int16)
v = np.float64(v)

d = np.frombuffer(d, np.int16)
d = np.float64(v)
print(u)

u=u[:len(d)]

# identification

f = pa.filters.FilterNLMS(n=4, mu=0.1, w="random")
y, e, w = f.run(d, u )

e = e.astype('int16')
wavfile.write(errorsignal, lfs, e)