import cv2
import numpy as np
import matplotlib.pyplot as plt

import thermal_monitor.config as config
from thermal_monitor.thermal_frame import ThermalFrame
import thermal_monitor.utils as utils

# path_mov = r"C:\Users\RDCunha\Documents\GitHub\CANDOR\thermal_monitoring\high_low_high_breathingl.mp4"
path_mov = r"U:\Users Common\NLamb\CANDOR\captures\ca_demo_v3_alex\ca_video_thermal.mp4"
count = 0 
frame_counter = 0
error_counter = 0
total_frame_counter = 500
start_frame_counter = 0
start_frame = 0
success = True
final_face = []
final_br = []
frame_flag = True

thermal_frame_queue = []
temperature_pool = {}
breath_rate_pool = {}
breath_curve_ax_pool = {}
timestamp = []
timestamp_frame = []

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

# def visualize_breath_curves(faces):
#     if plot_update_counter < config.BREATH_CURVE_UPDATE_FRAMES:
#         plot_update_counter += 1
#         return
#     plot_update_counter = 0
#     if not breath_curve_figure_state:
#         if breath_curve_figure is not None:
#             breath_curve_figure.clear()
#         breath_curve_figure = plt.figure()
#         plt.show(block=False)
#     if set([face.uuid for face in faces]) != set(breath_curve_ax_pool.keys()):
#         for key, value in breath_curve_ax_pool.items():
#             value.remove()
#         breath_curve_ax_pool = {}
#     for index, face in enumerate(faces):
#         if face.uuid not in breath_curve_ax_pool:
#             ax = breath_curve_figure.add_subplot(len(faces), 1, index + 1, label=face.uuid)
#             breath_curve_ax_pool[face.uuid] = ax
#         else:
#             ax = breath_curve_ax_pool[face.uuid]
#         ax.clear()
#         ax.plot(*face.breath_samples, c=utils.uuid_to_color(face.uuid, ub=1))
#         ax.set_title(face.uuid[:4])
#     plt.draw()
#     plt.pause(0.001)
#     return plot_update_counter

def plot(var):
    plt.plot(np.linspace(0, len(var), len(var)), var)
    plt.title("Respiration Rate")
    plt.xlabel("Frames")
    plt.ylabel("Breath Frequency (Units?)")
    plt.show()

# print("Checking video path")
try: 
    video = cv2.VideoCapture(path_mov)
except:
    print("Video does not exist")


while frame_counter < total_frame_counter:
    success, frame = video.read()

    if not success:
        error_counter += 1
        continue

    if start_frame > start_frame_counter:
        start_frame_counter += 1
        continue

    frame_counter += 1

    count += (1/60)
    timestamp.append(count)
    timestamp_frame.append(count)
    if frame_counter % config.MAX_CACHED_FRAMES == 0:
        timestamp_frame = []
    # print("Timestamp main:")
    # print(timestamp)

    print('Visualizing estimation result. Press Ctrl + C to stop.')
    # print("entered run method")
    thermal_frame = ThermalFrame(frame, timestamp_frame)
    # if thermal_frame._detect() is None:
    #      frame_flag = False
    if len(thermal_frame_queue) > 0:
        thermal_frame.link(thermal_frame_queue[-1])
    if len(thermal_frame_queue) >= config.MAX_CACHED_FRAMES:
        thermal_frame_queue.pop(0)
        thermal_frame_queue[0].detach()
    thermal_frame_queue.append(thermal_frame)
    annotation_frame = thermal_frame.thermal_frame
    visualize_bounding_boxes(annotation_frame, thermal_frame.thermal_faces)
    visualize_breath_rates(annotation_frame, thermal_frame.thermal_faces)
    for face in thermal_frame.thermal_faces:
        final_face.append(face.breath_samples[1][-1]) 
        # print("Breath samples")
        # print(*face.breath_samples)
        # if not frame_flag:
        #     print('No bounding box')
        #     try:
        #         final_face.append(final_face[-1])
        #     except:
        #         continue
        # else:
        #     final_face.append(face.breath_samples[1][-1]) 
        # visualize_breath_curves(thermal_frame.thermal_faces)
    # cv2.imshow('thermal monitoring', cv2.resize(annotation_frame, config.VISUALIZATION_RESOLUTION))
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # print("Frames completed", str(frame_counter))
    # frame_flag = True

# print(final_face)
print("Final rate")
print(final_br)
print(len(final_br))
# plot(final_face)
plot(final_br)
video.release()
cv2.destroyAllWindows()


print("completed")
# exit()
