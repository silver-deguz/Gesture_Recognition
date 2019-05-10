#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:26:33 2019

@author: Hike
"""

#!/usr/bin/env python3
 # -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:09:14 2019

@author: Hike
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage import measure
import os 

def pad(midpoint,width,image):
    
    if(mid_point + width > 640):
        print('r')
        extra_r   = mid_point + width - 640
        pad_r     = np.zeros((240, extra_r))
        new_im = np.concatenate((image[:,midpoint-width:],pad_r),axis  =1)
    elif(mid_point - width < 0):
        print('l')
        extra_l = width - mid_point 
        
        pad_l   = np.zeros((240,extra_l))
        new_im = np.concatenate((pad_l,image[:,:midpoint +width]),axis  =1)
    else:
        new_im =   image[:,midpoint-width:mid_point +width]  
        
    
    return new_im

base_path = '/Users/Hike/Desktop/Spring 2019/ECE 228/project/data/leapGestRecog/'
#base_path = '/Users/Hike/Desktop/Spring 2019/ECE 228/project/data/leapGestRecog/00/10_down/frame_00_10_0072.png'

# Find contours at a constant value of 0.8




width  = 120 + 20
os.chdir(base_path)

out = dict()
i   = 0
k = 0
for person in os.listdir():
    if not person.startswith('.'):
        path = base_path + person + '/'
        os.chdir(path)
        for gesture in os.listdir():
            if not gesture.startswith('.'):
                if (i == 0):
                    out[gesture]  = []
                    
                path1   = path + gesture + '/'
                os.chdir(path1)
                for each_im in os.listdir():
                    if not each_im.startswith('.'):
                        path2 = path1 + each_im
                       
                        r = plt.imread(path2)
                        contours = measure.find_contours(r, 0.2)
                        max_val = contours[0][:,1].max()
                        min_val = contours[0][:,1].min()
                        mid_point = int((max_val+min_val)/2)
                        if (each_im == 'frame_08_02_0095.png'):
                            print('here')
                        new_image = pad(mid_point,width,r)
                        out[gesture].append(new_image)
#                        if (k%500 == 0):
#                            fig, ax = plt.subplots()
#                            ax.imshow(new_image, interpolation='nearest', cmap=plt.cm.gray)
#                            plt.show()
                        k = k +1
                os.chdir('../')
        os.chdir('../')        
        i  = 2
os.chdir(base_path)
#%%
pickle.dump(out,open( "save.p", "wb" ) )


