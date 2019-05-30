import numpy as np
import pickle
import os 


def load():

    path = os.getcwd()
    temp = path
    path = path + '/data/'
    os.chdir(path)
    data = []
    labels = []
    directories = []
    for out_files in os.listdir():
        
        if out_files == 'Gesture_Recognition.ipynb':
            continue
        if not out_files.startswith('.'):
            print('Reading file ',out_files,'...')
            if (out_files == 'output_lb.p'):
                read_file =  path + out_files
                labels = pickle.load( open( read_file, "rb" ) )
                continue
            directories.append(out_files)
            directories.sort(key=lambda x: int(x[10:12]) if len(x)==14 else int(x[10:11]))
    for i in directories:
        read_file =  path  + i
        data_t = pickle.load( open( read_file, "rb" ) )    
        data.append(data_t)
        
    directories.sort()
 
    ln = len(data)
    im_ln = len(data[0][0])
    M     = len(data[0])
    label = np.array(labels[0:ln*M])
    dataa = np.zeros((M*ln,im_ln,im_ln,1))
    l = 0
    for i in data:
       for j in i:
            dataa[l,:,:,0] = j
            l = l +1
    os.chdir(temp)
    return   dataa, label    
