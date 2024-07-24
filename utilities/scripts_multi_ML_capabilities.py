import os
import cv2
import mediapipe as mp
import pandas as pd
from PIL import Image, ImageDraw,ImageFont
from pytube import YouTube
import numpy as np
from keras.models import load_model
import tempfile
import shutil
from joblib import load

# UTILITIES
def video_images(vid_url, images_folder, interval=30):
    #extract frames from input video using pytube
    yt = YouTube(vid_url)
    video_stream = yt.streams.get_highest_resolution()

    #create a temporary file to save the video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    #download the video to the temporary file
    video_stream.download(filename=temp_file)

    #use OpenCV to capture frames every interval seconds
    cap = cv2.VideoCapture(temp_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #check if the current frame is at the desired interval
        if frame_count % (fps * interval) == 0:
            output_filename = f"frame_{frame_count // fps}.jpg"
            frame_path = os.path.join(images_folder, output_filename)
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()

    #remove the temporary video file
    os.remove(temp_file)

    return images_folder

# UTILITIES
def process_images_and_landmarks(input_folder_path, output_folder_path, image_output):
    #function takes folder path from video_images function, processes the landmark coordinates of images
    #creates an image with the wireframe and a csv containing all landmark coordinates
    #output will be the folder path for wireframe images and complete path to the csv file
    
    #create list of landmark pose names
    landmark_names=['nose_x', 'nose_y', 'nose_z', 'left_eye_inner_x', 'left_eye_inner_y', 'left_eye_inner_z', \
                'left_eye_x', 'left_eye_y', 'left_eye_z','left_eye_outer_x', 'left_eye_outer_y', 'left_eye_outer_z', \
                'right_eye_inner_x', 'right_eye_inner_y', 'right_eye_inner_z','right_eye_x', 'right_eye_y', 'right_eye_z', \
                'right_eye_outer_x', 'right_eye_outer_y', 'right_eye_outer_z','left_ear_x', 'left_ear_y', 'left_ear_z', \
                'right_ear_x', 'right_ear_y', 'right_ear_z', 'mouth_left_x', 'mouth_left_y', 'mouth_left_z',\
                'mouth_right_x', 'mouth_right_y', 'mouth_right_z', 'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', \
                'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 'left_elbow_x', 'left_elbow_y', 'left_elbow_z', \
                'right_elbow_x', 'right_elbow_y', 'right_elbow_z', 'left_wrist_x', 'left_wrist_y', 'left_wrist_z',\
                'right_wrist_x', 'right_wrist_y','right_wrist_z', 'left_pinky_x', 'left_pinky_y', 'left_pinky_z', \
                'right_pinky_x', 'right_pinky_y', 'right_pinky_z', 'left_index_x', 'left_index_y', 'left_index_z', \
                'right_index_x', 'right_index_y', 'right_index_z', 'left_thumb_x', 'left_thumb_y', 'left_thumb_z', \
                'right_thumb_x', 'right_thumb_y', 'right_thumb_z', 'left_hip_x', 'left_hip_y', 'left_hip_z', \
                'right_hip_x', 'right_hip_y', 'right_hip_z', 'left_knee_x', 'left_knee_y', 'left_knee_z', \
                'right_knee_x', 'right_knee_y', 'right_knee_z', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z', \
                'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'left_heel_x', 'left_heel_y', 'left_heel_z', \
                'right_heel_x', 'right_heel_y', 'right_heel_z', 'left_foot_index_x', 'left_foot_index_y', 'left_foot_index_z', \
                'right_foot_index_x', 'right_foot_index_y', 'right_foot_index_z']

    #empty dict to populate with x,y,z coordinates for each pose
    landmark_dict = {'pose_name': []}
    yoga_pose = "empty" #no name for test data

    for img_file in os.listdir(input_folder_path):
        img_path = os.path.join(input_folder_path, img_file)
        img = Image.open(img_path) 
        file = img_path 

        # MP prep
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
            
        with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True) as pose:
            image = cv2.imread(file)

            try:
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
                #check if landmarks were detected
                if results.pose_landmarks is None:
                    print("No landmarks detected in frame:", img_file)
                    continue

                #if landmarks are detected, then process and save the image
                annotated_image = image.copy()
                mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, \
                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                #save wireframe of each image in folder with pose name or 'empty' for new data
                file_pose = img_file.split('.')
                filename = file_pose[0] + '.jpg'
                saved_image_path = os.path.join(image_output, filename) 
                cv2.imwrite(saved_image_path, annotated_image)

            except cv2.error:
                print("ERROR: Source image is empty", img_file)
                continue       

        count = 0
        j = 0
        try:
            landmarks = results.pose_world_landmarks.landmark 
            landmark_dict['pose_name'].append(yoga_pose)

            #grabs the 3D landmark coordinates assigns to variable  
            while count < len(landmarks):
                for i in range(len(mp_pose.PoseLandmark)):
                    x = landmarks[i].x
                    y = landmarks[i].y
                    z = landmarks[i].z
                        
                    #populating dictionary with coordinates for each landmarks
                    landmark_pt_x = landmark_names[j] 
                    if landmark_pt_x in landmark_dict:
                        landmark_dict[landmark_pt_x].append(x)
                    else:
                        landmark_dict[landmark_pt_x] = [x]
                            
                    j += 1
                        
                    landmark_pt_y = landmark_names[j]
                    if landmark_pt_y in landmark_dict:
                        landmark_dict[landmark_pt_y].append(y)
                    else:
                        landmark_dict[landmark_pt_y] = [y]
                            
                    j += 1
                        
                    landmark_pt_z = landmark_names[j]
                    if landmark_pt_z in landmark_dict:
                        landmark_dict[landmark_pt_z].append(z)
                    else:
                        landmark_dict[landmark_pt_z] = [z]
                            
                    j += 1
                    count += 1

        except AttributeError:
            print("AttributeError: 'NoneType' object has no attribute 'landmark'", img_file)
            continue

    #prep and print data to csv
    csv_name = 'test image_data.csv'
    csv_path = os.path.join(output_folder_path, csv_name)
    data = pd.DataFrame(landmark_dict)
    #further process data
    data = adding_features(data)
    
    data.to_csv(csv_path)#, index=False)

    #return the list of saved image paths and the csv path
    return output_folder_path, csv_path

def calculate_angle(P1, P2, P3):
    #engineered features to enhance ML performace
    v1 = P2 - P1  #vector from P1 to P2
    v2 = P3 - P2  #vector from P2 to P3
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    angle = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    return np.degrees(angle)

def adding_features(data):
    #creating df of needed columns

    #removing unecessary landmarks
    core_landmarks = ['nose_x', 'nose_y', 'nose_z', \
                'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', \
                'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', \
                'left_elbow_x', 'left_elbow_y', 'left_elbow_z', \
                'right_elbow_x', 'right_elbow_y', 'right_elbow_z', \
                'left_wrist_x', 'left_wrist_y', 'left_wrist_z',\
                'right_wrist_x', 'right_wrist_y','right_wrist_z',
                'left_hip_x', 'left_hip_y', 'left_hip_z', \
                'right_hip_x', 'right_hip_y', 'right_hip_z',\
                'left_knee_x', 'left_knee_y', 'left_knee_z', \
                'right_knee_x', 'right_knee_y', 'right_knee_z', \
                'left_ankle_x', 'left_ankle_y', 'left_ankle_z', \
                'right_ankle_x', 'right_ankle_y', 'right_ankle_z']

    new_df = data[core_landmarks].copy()

    #adding engineered features to df
    #Shoulder Hip Knee Angle
    new_df.loc[:, 'left_shoulder_hip_knee_angle'] = new_df.apply(lambda row: calculate_angle(
        np.array([row['left_shoulder_x'], row['left_shoulder_y'], row['left_shoulder_z']]),
        np.array([row['left_hip_x'], row['left_hip_y'], row['left_hip_z']]),
        np.array([row['left_knee_x'], row['left_knee_y'], row['left_knee_z']])
    ), axis=1)

    new_df.loc[:, 'right_shoulder_hip_knee_angle'] = new_df.apply(lambda row: calculate_angle(
        np.array([row['right_shoulder_x'], row['right_shoulder_y'], row['right_shoulder_z']]),
        np.array([row['right_hip_x'], row['right_hip_y'], row['right_hip_z']]),
        np.array([row['right_knee_x'], row['right_knee_y'], row['right_knee_z']])
    ), axis=1)

    #Hip Knee Ankle Angle
    new_df.loc[:, 'left_hip_knee_ankle_angle'] = new_df.apply(lambda row: calculate_angle(
        np.array([row['left_hip_x'], row['left_hip_y'], row['left_hip_z']]),
        np.array([row['left_knee_x'], row['left_knee_y'], row['left_knee_z']]),
        np.array([row['left_ankle_x'], row['left_ankle_y'], row['left_ankle_z']])
    ), axis=1)

    new_df.loc[:, 'right_hip_knee_ankle_angle'] = new_df.apply(lambda row: calculate_angle(
        np.array([row['right_hip_x'], row['right_hip_y'], row['right_hip_z']]),
        np.array([row['right_knee_x'], row['right_knee_y'], row['right_knee_z']]),
        np.array([row['right_ankle_x'], row['right_ankle_y'], row['right_ankle_z']])
    ), axis=1)

    #Shoulder Elbow Wrist Angle
    new_df.loc[:, 'left_shoulder_elbow_wrist_angle'] = new_df.apply(lambda row: calculate_angle(
        np.array([row['left_shoulder_x'], row['left_shoulder_y'], row['left_shoulder_z']]),
        np.array([row['left_elbow_x'], row['left_elbow_y'], row['left_elbow_z']]),
        np.array([row['left_wrist_x'], row['left_wrist_y'], row['left_wrist_z']])
    ), axis=1)

    new_df.loc[:, 'right_shoulder_elbow_wrist_angle'] = new_df.apply(lambda row: calculate_angle(
        np.array([row['right_shoulder_x'], row['right_shoulder_y'], row['right_shoulder_z']]),
        np.array([row['right_elbow_x'], row['right_elbow_y'], row['right_elbow_z']]),
        np.array([row['right_wrist_x'], row['right_wrist_y'], row['right_wrist_z']])
    ), axis=1)

    #Hip Shoulder Elbow Angle
    new_df.loc[:, 'left_hip_shoulder_elbow_angle'] = new_df.apply(lambda row: calculate_angle(
        np.array([row['left_hip_x'], row['left_hip_y'], row['left_hip_z']]),
        np.array([row['left_shoulder_x'], row['left_shoulder_y'], row['left_shoulder_z']]),
        np.array([row['left_elbow_x'], row['left_elbow_y'], row['left_elbow_z']])
    ), axis=1)

    new_df.loc[:, 'right_hip_shoulder_elbow_angle'] = new_df.apply(lambda row: calculate_angle(
        np.array([row['right_hip_x'], row['right_hip_y'], row['right_hip_z']]),
        np.array([row['right_shoulder_x'], row['right_shoulder_y'], row['right_shoulder_z']]),
        np.array([row['right_elbow_x'], row['right_elbow_y'], row['right_elbow_z']])
    ), axis=1)

    #Head Shoulder Hip Angle
    new_df.loc[:, 'left_head_shoulder_hip_angle'] = new_df.apply(lambda row: calculate_angle(
        np.array([row['nose_x'], row['nose_y'], row['nose_z']]),
        np.array([row['left_shoulder_x'], row['left_shoulder_y'], row['left_shoulder_z']]),
        np.array([row['left_hip_x'], row['left_hip_y'], row['left_hip_z']])
    ), axis=1)

    new_df.loc[:, 'right_head_shoulder_hip_angle'] = new_df.apply(lambda row: calculate_angle(
        np.array([row['nose_x'], row['nose_y'], row['nose_z']]),
        np.array([row['right_shoulder_x'], row['right_shoulder_y'], row['right_shoulder_z']]),
        np.array([row['right_hip_x'], row['right_hip_y'], row['right_hip_z']])
    ), axis=1)

    return new_df

# UTILITIES
def run_model(new_data, model_paths):
    # function for different models

    predictions = {}

    for key, value in model_paths.items():
        predictions[key] = None
        
        if key == 'Keras':
            model_path = value 
            model = load_model(model_path)
            predictions[key] = model.predict(new_data)
            
        if key == 'RFC':
            model_path = value 
            model = load(model_path)
            predictions[key] = model.predict_proba(new_data)

        if key == 'LR':
            model_path = value 
            model = load(model_path)
            predictions[key] = model.predict_proba(new_data)
             
    return predictions

# UTILITIES
def extract_prediction(csv_image_data):
    #function takes in the csv file returned by the process_images_and_landmarks function (which should be saved as two variables)
    #and returns a list containing the predicted values

    df = pd.read_csv(csv_image_data)

    data = df.values
    features_data = data[:, 1:]
    
    new_data = np.asarray(features_data).astype(np.float32)

    class_names = ['cat pose', 'chair pose', 'corpse pose', 'cow pose', 'downward dog pose',\
                   'flat back pose', 'high lunge pose', 'knee to elbow plank pose',\
                   'knees to chest pose', 'low lunge pose', 'mountain pose',\
                   'runners lunge twist pose', 'seated spinal twist pose', 'side plank yoga pose',\
                   'standing forward bend pose', 'tabletop pose', 'three legged dog pose', \
                   'tip toe pose', 'tree pose', 'upper plank pose']

    predictions_dict = {} #empty holds predicted pose names or 'unknown'

    #a list or dict of model paths 
    model_paths = {'Keras': r'PATH\Model5_Keras.h5',
                   'RFC': r'PATH\Model4_RFC.joblib',
                   'LR': r'PATH\Model2_LogisticRegression.joblib'}
    
    predictions = run_model(new_data, model_paths) #returns dictionary with model as key and array/rows as values

    for key, value in predictions.items():
        predictions_list = [] #this should reset for every key
 
        for i in range(len(value)):
            arr = value[i] #takes the first array/row of results
            formatted_percentage = [round(value * 100) for value in arr]
            max_val = max(formatted_percentage)
            max_index = formatted_percentage.index(max_val) 
            print(key, " ", max_val)

            if max_val >= 90:
                predicted_label = class_names[max_index]
                predictions_list.append(predicted_label)           

            else:
                predicted_label = "unknown"
                predictions_list.append(predicted_label)

        predictions_dict[key] = predictions_list #this should append the full list to the correct model and predicted poses from model.
    
    return predictions_dict #this should return all models and a list containing each prediction per model

# UTILITIES
def image_write(image_path, predicted_pose, for_output):
    #function takes path to wireframe images and writes predicted name on them

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('arial.ttf', 20)
    draw.text((5, 5), predicted_pose, font=font, fill='red')

    image.save(for_output)

image_folder = #FOLDER WHERE FRAMES WILL BE PLACED
os.makedirs(image_folder, exist_ok=True)

#SAMPLE YOGA URLS
vid_url = r"https://www.youtube.com/watch?v=Ho9em79_0qg&t=60s" 
#vid_url = r"https://www.youtube.com/watch?v=dXGCALvZSZ0" #test url
#vid_url = r'https://www.youtube.com/watch?v=S5lH_B2K-X8&t=60s'
print('program started')

frame_folder = video_images(vid_url, image_folder) 

#CREATE FOLDERS AND INPUT DIRECTORIES BELOW
for_output = #FOLDER FOR MP RESULTS CSV FILE AND FOLDER FOR WIREFRAME IMAGES
os.makedirs(for_output, exist_ok=True)
image_output = #FOLDER FOR MP RESULTS WIREFRAME IMAGES
os.makedirs(image_output, exist_ok=True)

images_path, csv_file_path = process_images_and_landmarks(image_folder, for_output, image_output)

list_of_predictions = extract_prediction(csv_file_path) #dictionary of model/predct

for key in list_of_predictions.keys():
    count = 0 #initialize count for index in values list from dict_predicts
    folder_name = key + '_model_image_output' #this created inside wireframe

    new_folder_path = os.path.join(image_output, folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    
    for filename in os.listdir(image_output):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_file_path = os.path.join(image_output, filename)
            output_file_path = os.path.join(new_folder_path, filename)
            image_write(input_file_path, list_of_predictions[key][count], output_file_path)
            
        count += 1
    
print("yoga program complete")
    
