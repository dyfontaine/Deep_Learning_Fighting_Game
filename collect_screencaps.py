from PIL import ImageGrab
import numpy as np
import cv2
import win32api
import time
import os
import csv

#function to pad an integer with 0's for file naming
#ex: 1 becomes '001'. 22 becomes '022'
def format_number(x):
    return str(x).zfill(4)

#%%

#calculate game number and create directory to store images
folder_list = os.listdir('screencaps_fighting')
current_max = 0
if len(folder_list) > 0:
    current_max = int(folder_list[-1][-4:])
game_number = current_max + 1
os.makedirs('screencaps_fighting/game_' + format_number(game_number))

#wait a few seconds so that user can start game
print('waiting...')
time.sleep(6)

#Capture 600 sequential screenshots
print('collecting screenshots...')
img_list = []
labels = []
for i in range(0,500):
    #take screenshot and add to list
    img = ImageGrab.grab(bbox=(0,0,1366,768)) 
    img_list.append(img)
    
    #check which keys are pressed
    left = win32api.GetAsyncKeyState(65) # A Key
    #up = win32api.GetAsyncKeyState(38)
    right = win32api.GetAsyncKeyState(68) # D key
    #down = win32api.GetAsyncKeyState(40)
    #assign class label and add to list
    jKey = win32api.GetAsyncKeyState(74)
    if (left <= -32767 or left==1):
        my_label = 9
    elif (right <= -32767 or right==1):
        my_label = 10
    elif (jKey <= -32767 or jKey==1):
        my_label = 11
    else:
        my_label = 0
    labels.append(my_label)

print('done collecting. now saving...')

#Save screenshots to 'screencaps' folder
j=0
for img in img_list:
    j += 1
    #frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    img_np = np.array(img)
    img_cropped = img_np[80:730,194:1170]
    cv2.imwrite('screencaps_fighting/game_' + format_number(game_number) + '/pic_' + format_number(j) + '.jpg', img_cropped)
#save labels to file
with open('screencaps_fighting/game_' + format_number(game_number) + '/labels.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for l in labels:
        writer.writerow([l])
print('done saving')
    

#%%
#Show the size of variables stored in memory
#from sys import getsizeof
#print(getsizeof(img_list))
#print(getsizeof(img_list[1]))

#%%
#Show an image
#import matplotlib.pyplot as plt
#time.sleep(3)

#img = np.array(ImageGrab.grab(bbox=(0,0,1366,768)))
#img_cropped = img[80:730,194:1170]

#plt.imshow(img_cropped, cmap = 'gray')





