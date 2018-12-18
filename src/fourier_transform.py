import wave
import math
import numpy as np
import contextlib


def running_mean(x, window_size):
    cumulative_sum = np.cumsum(np.insert(x, 0, 0))
    return (cumulative_sum[window_size:] - cumulative_sum[:-window_size])/window_size


def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved=True):
    if sample_width == 1:
        data_type = np.uint8  # unsigned char
    elif sample_width == 2:
        data_type = np.int16  # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels_s = np.fromstring(raw_bytes, dtype=data_type)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels_s.shape = (n_frames, n_channels)
        channels_s = channels_s.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels_s.shape = (n_channels, n_frames)
    return channels_s


def low_pass_filter(cut_off_frequency, audio_file_location):
    with contextlib.closing(wave.open(audio_file_location, 'rb')) as spf:
        sample_rate = spf.getframerate()
        amplitude_width = spf.getsampwidth()
        number_of_channels = spf.getnchannels()
        number_of_frames = spf.getnframes()

        # Extract Raw Audio from multi-channel wave file
        signal = spf.readframes(number_of_frames*number_of_channels)
        spf.close()
        channels = interpret_wav(signal, number_of_frames, number_of_channels, amplitude_width, True)

        # Obtain window size
        frequency_ratio = (cut_off_frequency/sample_rate)

        # Use moving average (only on first channel)
        filtered = running_mean(channels[0], int(math.sqrt(0.196196 + frequency_ratio**2)/frequency_ratio)).astype(channels.dtype)

        wav_file = wave.open('filtered.wav', "w")
        wav_file.setparams((1, amplitude_width, sample_rate, number_of_frames, spf.getcomptype(), spf.getcompname()))
        wav_file.writeframes(filtered.tobytes('C'))
        wav_file.close()


def generate_raw_audio(audio_file_location):
    with contextlib.closing(wave.open(audio_file_location, 'rb')) as spf:
        amplitude_width = spf.getsampwidth()
        number_of_channels = spf.getnchannels()
        number_of_frames = spf.getnframes()

        # Extract Raw Audio from multi-channel wave file
        signal = spf.readframes(number_of_frames*number_of_channels)
        spf.close()
        return interpret_wav(signal, number_of_frames, number_of_channels, amplitude_width, True)



import os

podcast_file_loc = r"C:\Users\Samuel\Desktop\selective_hearing\wav_data"
podcast_combined_file_loc = r"C:\Users\Samuel\Desktop\selective_hearing\combined_noise_audio/"

noise_data = os.listdir(podcast_combined_file_loc)




training_matrix = list()
expected_values = list()

noise_data_itr = 0

def average_convolution(data, convolve_size=50):
    """ Peform an average convolution """
    data = data.tolist()
    new_arr = list()
    for k in range(len(data)//convolve_size - 1):
        a = data[k*convolve_size:(k+1)*convolve_size]
        new_arr.append(sum(a)/convolve_size)
    return np.array(new_arr)

test = noise_data[2000:2500]
noise_data = noise_data[:2000]

expected_testfreq = list()
testm = list()

for i in test:
    try:
        audio_title = i.split("&")
        expected_testfreq.append(int(audio_title[0].split("_")[1].replace(".wav", "")))
        #actual_file = "{}{}".format(podcast_file_loc, audio_title[1])
        noise_file = "{}{}".format(podcast_combined_file_loc, i)
        noise = generate_raw_audio(noise_file)
        testm.append(average_convolution(noise[0]))

    except Exception:
        pass

for i in noise_data:
    try:
        audio_title = i.split("&")
        expected_freq = int(audio_title[0].split("_")[1].replace(".wav", ""))
        #actual_file = "{}{}".format(podcast_file_loc, audio_title[1])
        noise_file = "{}{}".format(podcast_combined_file_loc, i)
        noise = generate_raw_audio(noise_file)
        training_matrix.append(average_convolution(noise[0]))
        expected_values.append(expected_freq)
        print(noise_data_itr, len(noise_data))
        noise_data_itr+=1
    except Exception:
        pass
    #actual = generate_raw_audio(actual_file)
    #min_wav_length = min(len(noise[0]), len(actual[0]))
    #noise = noise[0][:min_wav_length]
    #actual = actual[0][:min_wav_length]
    #print(noise[:10])
    #print(actual[:10])




import pickle
from sklearn.neural_network import multilayer_perceptron



### ARTIFICIAL NEURAL NETWORK

low_pass_net = multilayer_perceptron.MLPRegressor()
low_pass_net.fit(training_matrix, expected_values)
with open("low_pass_net.scikit", 'wb') as f:
    pickle.dump(low_pass_net, f)

avg = 0
for i in range(len(test)):
    avg += abs(low_pass_net.predict([testm[i]])[0] - expected_testfreq[i])

print(avg/len(training_matrix))  # ANN error

### SUPPORT VECTOR MACHINE

from sklearn.svm import SVR

support_vector = SVR(kernel='rbf')
support_vector.fit(training_matrix, expected_values)

avg = 0
for i in range(len(test)):
    avg += abs(support_vector.predict([testm[i]])[0] - expected_testfreq[i])

print(avg/len(training_matrix))  # SVM error


### CONVOLUATIONAL NEURAL NETWORK


import keras
from keras.models import Sequential,Input,Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

epochs = 100
batch_size = 50

audio_convolutional_net = Sequential()
audio_convolutional_net.add(Conv2D(32, kernel_size=(3, 1), activation='linear', input_shape=(45655,1,1), padding='same'))
audio_convolutional_net.add(LeakyReLU(alpha=0.1))
audio_convolutional_net.add(MaxPooling2D((2, 1), padding='same'))
audio_convolutional_net.add(Conv2D(64, (3, 1), activation='linear', padding='same'))
audio_convolutional_net.add(LeakyReLU(alpha=0.1))
audio_convolutional_net.add(MaxPooling2D(pool_size=(2, 1), padding='same'))
audio_convolutional_net.add(Conv2D(128, (3, 1), activation='linear', padding='same'))
audio_convolutional_net.add(LeakyReLU(alpha=0.1))
audio_convolutional_net.add(MaxPooling2D(pool_size=(2, 1), padding='same'))
audio_convolutional_net.add(Flatten())
audio_convolutional_net.add(Dense(128, activation='linear'))
audio_convolutional_net.add(LeakyReLU(alpha=0.1))
audio_convolutional_net.add(Dense(100, activation='softmax'))

audio_train = audio_convolutional_net.fit(training_matrix, expected_values, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(testm, expected_testfreq))

model_evauluation = audio_convolutional_net.evaluate(testm, expected_testfreq, verbose=0)

print(model_evauluation[0], model_evauluation[1]) # test loss and accuracy





