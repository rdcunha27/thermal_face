import h5py
import numpy as np
import time
import os
import cv2

def file_feed(path):
    """ Returns an iterator of the frames and timestamps from a recording file.

    Args:
        path: The path of the recording file.
    """
    # cap = cv2.VideoCapture(r"U:\Users Common\NLamb\CANDOR\vitals_video_trimmed_cropped.mp4")
    # frameState, frame = cap.read()
    # count = 0 #debug with length of video
    print("ENTERED")
    os.chdir("file_processing")
    print("processing file")
    # while frameState:
    #     cv2.imwrite("frame%d.jpg" % count, frame)     # save frame as JPEG file
    #     frameState, frame = cap.read()
    #     count += 1
    # cap.release()

    # for image in os.getcwd():
    #     yield np.array(image['{}/raw_frame'.format(image)]), np.array(image['{}/timestamp'.format(image)])[0]

    cwd = os.getcwd()
    print(cwd)
    for filename in os.listdir(cwd):
        if filename.endswith(".jpg"):
            image = cv2.imread(filename)
            raw_frame = np.array(image)
            # timestamp = filename.split(".")[0]  # Assuming the timestamp is extracted from the filename
            print(filename)
            yield raw_frame

# def stream_feed(gige_camera_id):
#     """ Returns an iterator of the frames and timestamps from GIGE thermal camera.

#     Args:
#         gige_camera_id: The id of the GIGE thermal camera that streams the input.
#     """
#     import matlab.engine
#     matlab_engine = matlab.engine.start_matlab()
#     matlab_engine.addpath(os.path.dirname(os.path.realpath(__file__)))
#     gigecam = matlab_engine.init_gigecam(gige_camera_id)
#     while True:
#         raw_frame, timestamp = matlab_engine.get_temperature(gigecam), time.time()
#         frame = np.array(raw_frame._data).reshape(raw_frame.size, order='F')
#         yield frame, timestamp
