__author__ = "Erik Seetao, Hayk Hovhannisyan"

import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.measure import find_contours

import os 


class HandsDataLoader:
    def __init__(self,path,newdirectory):
        self.path = path
        self.width = 120 #height is 240, so plus minus width on both sides gives 240 as square image
        self.newdirectory = newdirectory


    def crop_img(self, mid_point, image):
        '''
        Checks if image needs padding, then crops image to 240x240
        '''
        
        #padding image
        if(mid_point + self.width > 640):
            print('right padding...')
            extra_r = mid_point + self.width - 640
            pad_r   = np.zeros((240, extra_r))
            new_im = np.concatenate((image[:,mid_point-self.width:],pad_r),axis  =1)

        elif(mid_point - self.width < 0):
            print('left padding...')
            extra_l = self.width - mid_point 
            
            pad_l   = np.zeros((240,extra_l))
            new_im  = np.concatenate((pad_l,image[:,:mid_point+self.width]),axis  =1)
        #image does not need to be padded
        else:
            new_im = image[:,mid_point-self.width:mid_point+self.width]  
            
        return new_im

    def rotate_image(self, image, angle):
        '''
        Rotates the image counterclockwise by angle in degrees
        '''
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        return result

    def process_img(self):
        '''
        Data preprocessing on images and saves them to appropriate dataset folders
        '''

        print("list dir is", os.listdir(self.path+'/leapGestRecog')) #prints out user list


        for person in os.listdir(self.path+'/leapGestRecog'):
            user_path = self.path+'/leapGestRecog/' + person + '/'
            print("userpath is:",user_path)
            print("person is:",person)
            
            for gesture in os.listdir(user_path):
                gesture_path = user_path + gesture
                print("gesture path is:",gesture_path)

                img_counter = 1
                class_counter = 1

                #checkpoint
                checkpoint_path = self.path + '/' + self.newdirectory + '/' + person + '/' + gesture
                if len(os.listdir(checkpoint_path)) == 200: #if processed 200 images
                    print("already finished processing gesture ",gesture," for user ",person)
                    continue
                else:
                    for img in os.listdir(gesture_path):

                        print("processing... ", gesture_path+'/'+img) #is absolute path of image
                        
                        r = plt.imread(gesture_path + '/' + img)

                        contours = find_contours(r, 0.35) #find hand

                        # #plotting contours
                        # fig, ax = plt.subplots()
                        # ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)

                        # for n, contour in enumerate(contours):
                        #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
                        # ax.axis('image')
                        # ax.set_xticks([])
                        # ax.set_yticks([])
                        # #plt.show()
                        # ################

                        #find midpoint of hand
                        max_val = contours[0][:,1].max()
                        min_val = contours[0][:,1].min()
                        mid_point = int((max_val+min_val)/2)

                        # #to find vertical, not needed
                        # vmax_val = contours[0][:,0].max()
                        # vmin_val = contours[0][:,0].min()
                        # vmid_point = int((vmax_val+vmin_val)/2)

                        # # prints marker in center of hand
                        # #plt.figure()
                        # plt.imshow(r,cmap=plt.cm.gray)
                        # plt.plot(mid_point,vmid_point, marker = "+", color = 'r')
                        # plt.show()

                        cropped_img = self.crop_img(mid_point,r)

                        #rotated_img = self.rotate_image(cropped_img,-15)

                        # plt.figure()
                        # f, ax = plt.subplots(2, sharey=True)
                        # ax[0].imshow(cropped_img, cmap=plt.cm.gray)
                        # ax[1].imshow(rotated_img, cmap=plt.cm.gray)
                        # plt.show()

                        img_index = str(img_counter).zfill(4)
                        class_index = str(class_counter).zfill(2)
                        new_dir_save = self.path + '/' + self.newdirectory + '/' + person + '/' + gesture + '/'

                        plt.imshow(cropped_img, cmap=plt.cm.gray) 
                        plt.savefig(new_dir_save + "frame_" + person + "_" + class_index + "_" + img_index + ".png")
                        plt.close()

                        img_counter += 1
                        class_counter += 1



    def prep_folder(self):
        '''
        Check if new directory exists to save images in, otherwise builds the dataset files
        '''
        if not os.path.exists(self.newdirectory):
            os.makedirs(self.newdirectory)

            users = ['00','01','02','03','04','05','06','07','08','09']
            gestures = ['01_palm','02_l','03_fist','04_fist_moved','05_thumb','06_index','07_ok','08_palm_moved','09_c','10_down']
            #print(self.path + '/' + self.newdirectory + '/')
            for user in users:
                os.makedirs(os.path.join(self.path + '/' + self.newdirectory + '/', user))
                for gesture in gestures:
                    os.makedirs(os.path.join(self.path + '/' + self.newdirectory + '/' + user + '/', gesture))




if __name__ == '__main__':

    path = os.path.dirname(os.path.abspath(__file__)) #/Users/eseetao/Documents/Code
    #print(path)

    data = HandsDataLoader(path,"leapGestAug")
    data.prep_folder()
    data.process_img()