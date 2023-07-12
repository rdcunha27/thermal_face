import os
import cv2
import glob

directory = os.fsencode(r"U:\Users Common\NLamb\CANDOR\captures\thermal_test\thermal")

frame_list = []
frames = [cv2.imread(file) for file in glob.glob(r"U:\Users Common\NLamb\CANDOR\captures\thermal_test\thermal\*.jpg")]

for img in glob.glob(r"U:\Users Common\NLamb\CANDOR\captures\thermal_test\thermal\*.jpg"):
    frame_list.append(cv2.imread(img))
    print(cv2.imread(img))

# for i in os.listdir(directory):
#     cv2.imread(i)
#     frame_list.append(i)
print(frame_list)
exit(0)
for i in frame_list:
    cv2.imshow("Frame", i)

