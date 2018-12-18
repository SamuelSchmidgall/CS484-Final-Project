#import pickle
#from sklearn.neural_network import multilayer_perceptron


#with open("low_pass_net.scikit", 'rb') as f:
#    mlp = pickle.load(f)

#mlp.predict()

import os
import subprocess

for i in os.listdir(r'C:\Users\Samuel\Desktop\selective_hearing\audio-podcasts'):
    try:
        subprocess.call(['ffmpeg', '-i', os.path.realpath(r'C:/Users/Samuel/Desktop/selective_hearing/audio-podcasts/{}'.format(i)),
                         r'C:\Users\Samuel\Desktop\selective_hearing\wav_data/{}.wav'.format(i)])
    except Exception as e:
        print(e)



















