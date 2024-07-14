import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor


show_number = input("Enter the show number: ")
num_frames = int(input("Enter the number of frames to extract from each video: "))
start_time_minutes = int(input("Enter the starting time in minutes to begin capturing frames: "))


start_time_seconds = start_time_minutes * 60

input_folder_path = os.path.dirname(os.path.abspath(__file__))  
output_folder_path = 'output'
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


video_filenames = [f for f in os.listdir(input_folder_path) if f.endswith('.mkv')]

def extractFrames(filename, show_number, output_folder_path, num_frames, start_time_seconds):
    vidcap = cv2.VideoCapture(filename)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    start_frame = int(fps * start_time_seconds)  


    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    count = 0
    while count < num_frames:
        success, image = vidcap.read()
        if not success:
            break
        
        
        cv2.imwrite(os.path.join(output_folder_path, f"show{show_number}_Frame{count + 1}.png"), image)
        count += 1

    vidcap.release()  


start_time = time.perf_counter()


with ThreadPoolExecutor() as executor:
    futures = []
    
    
    for video_filename in video_filenames:
        
        future = executor.submit(extractFrames, 
                                  os.path.join(input_folder_path, video_filename), 
                                  show_number, 
                                  output_folder_path, 
                                  num_frames, 
                                  start_time_seconds)
        futures.append(future)

    
    for future in futures:
        future.result()  


end_time = time.perf_counter()


print("Elapsed time:", end_time - start_time)
