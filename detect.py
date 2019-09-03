from imageai.Detection import VideoObjectDetection
import os
import cv2
OUT_FOLDER="output"
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('{}/output.mp4'.format(OUT_FOLDER), fourcc, 2, (640, 480),0)
camera = cv2.VideoCapture(0)
print(camera.isOpened())

def forFrame(frame_number, output_array, output_count, returned_frame):
    print("Frame Number : ", frame_number)
    grayFrame = cv2.cvtColor(returned_frame, cv2.COLOR_BGR2GRAY)
    out.write(grayFrame)


execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.loadModel()

custom_objects = detector.CustomObjects(person=True, cell_phone=True)



detections = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects, camera_input=camera,
                                                   frame_detection_interval=1,
                                                   minimum_percentage_probability=30,
                                                   return_detected_frame=True, frames_per_second=20,
                                                   per_frame_function=forFrame,
                                                   display_percentage_probability=False, display_object_name=False,
                                                   save_detected_video=False)
camera.release()
out.release()

# print(video_path)
