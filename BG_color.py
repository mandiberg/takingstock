import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

file="C:/Users/jhash/Pictures/test/test2.jpg"


change_background_mp = mp.solutions.selfie_segmentation
change_bg_segment = change_background_mp.SelfieSegmentation()

sample_img = cv2.imread(file)
result = change_bg_segment.process(sample_img[:,:,::-1])
mask=np.repeat((1-result.segmentation_mask)[:, :, np.newaxis], 3, axis=2)
masked_img=mask*sample_img[:,:,::-1]/255 ##RGB format
# Identify black pixels where R=0, G=0, B=0
black_pixels_mask = np.all(masked_img == [0, 0, 0], axis=-1)
# Filter out black pixels and compute the mean color of the remaining pixels
mean_color = np.mean(masked_img[~black_pixels_mask], axis=0)[np.newaxis,np.newaxis,:] # ~ is negate
hue=cv2.cvtColor(mean_color, cv2.COLOR_RGB2HSV)[0,0,0]
lum=cv2.cvtColor(mean_color, cv2.COLOR_RGB2LAB)[0,0,0]

print("hue=",hue,"Luminosity=",lum)