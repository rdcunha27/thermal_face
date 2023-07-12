import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

import respiratory_config as config
from respiratory_thermal_frame import ThermalFrame
import respiratory_utils as utils

def calc_breath_rate(frames, root_direc, frame_rate=60):

    w, h = [512, 640]
    if frame_rate != 60:
        config.MAX_FPS = frame_rate
    config.SPLINE_SAMPLE_INTERVAL = 1/frame_rate
    count = 0 
    frame_counter = 0
    error_counter = 0
    total_frame_counter = 300
    final_face = []
    final_br = []

    thermal_frame_queue = []
    breath_rate_pool = {}
    breath_curve_ax_pool = {}

    timestamp_frame_cache = []

    def plot(var):
        plt.plot(np.linspace(0, len(var), len(var)), var)
        plt.title("Respiration Rate")
        plt.xlabel("Frames")
        plt.ylabel("Breath Frequency (Units?)")
        plt.show()

    def visualize_bounding_boxes(annotation_frame, faces):
            for face in faces:
                cv2.rectangle(
                    annotation_frame,
                    tuple(face.bounding_box[:2]),
                    tuple(face.bounding_box[2:]),
                    utils.uuid_to_color(face.uuid, mode='bgr'),
                    1
                )
                cv2.putText(
                    annotation_frame,
                    face.uuid[:2],
                    (face.bounding_box[0], face.bounding_box[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    utils.uuid_to_color(face.uuid, mode='bgr'),
                    1
                )

    def visualize_breath_rates(annotation_frame, faces):
            face_uuids = [face.uuid for face in faces]
            keys = [*breath_rate_pool.keys()]
            for key in keys:
                if key not in face_uuids:
                    breath_rate_pool.pop(key, None)
            for face in faces:
                if face.uuid not in breath_rate_pool or breath_rate_pool[face.uuid][0] >= config.BREATH_RATE_UPDATE_FRAMES:
                    breath_rate = face.breath_rate
                    if breath_rate is None:
                        return
                    breath_rate_pool[face.uuid] = [0, breath_rate]
                else:
                    breath_rate = breath_rate_pool[face.uuid][1]
                    final_br.append(round((breath_rate * 60), 3))
                    breath_rate_pool[face.uuid][0] += 1
                cv2.putText(
                    annotation_frame,
                    str(breath_rate * 60)[:5] + ' bpm',
                    (face.bounding_box[0], face.bounding_box[3] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    utils.uuid_to_color(face.uuid, mode='bgr'), 
                    1
                )

    def enhance_contrast(image_8):
        lab = cv2.cvtColor(image_8, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    while frame_counter < total_frame_counter:
        print("Frame: ", str(frame_counter))
        frame_8 = cv2.imread(os.path.join(root_direc, frames[frame_counter]))
        frame_8 = enhance_contrast(frame_8)

        count += (1/frame_rate)
        timestamp_frame_cache.append(count)

        if frame_counter % config.MAX_CACHED_FRAMES == 0:
            timestamp_frame_cache = []

        thermal_frame = ThermalFrame(frame_8, timestamp_frame_cache, w, h)

        if len(thermal_frame_queue) > 0:
            thermal_frame.link(thermal_frame_queue[-1])
        if len(thermal_frame_queue) >= config.MAX_CACHED_FRAMES:
            thermal_frame_queue.pop(0)
            thermal_frame_queue[0].detach()
        thermal_frame_queue.append(thermal_frame)
        annotation_frame = thermal_frame.thermal_frame
        visualize_bounding_boxes(frame_8, thermal_frame.thermal_faces)
        visualize_breath_rates(frame_8, thermal_frame.thermal_faces)
        for face in thermal_frame.thermal_faces:
            final_face.append(face.breath_samples[1][-1]) 

        frame_counter += 1

    cv2.destroyAllWindows()

    placeholder = final_br[0]
    while len(final_br) != len(frames):
         final_br.insert(placeholder)
         
    plot(final_br)
    return final_br #list containing bpm

root_direc = r"U:\Users Common\NLamb\CANDOR\captures\thermal_test\thermal"

frame_list = []
for i in os.listdir(root_direc):
    frame_list.append(i)
calc_breath_rate(frame_list, root_direc)