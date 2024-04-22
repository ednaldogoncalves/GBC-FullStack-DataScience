"""
Final Project: Suspect detection CCTV (closed-circuit television)
Final Project - AASD4016: Full Stack Data Science
Professor: Vejey Gandyer
Final Project - ASSD4017: Data Science Driven
Professor: Moe Fadaee
Student ID: 101373529
Student Name: Gonçalves, Ednaldo
"""

# Import Libraries
# import the necessary packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import face_recognition
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from imgbeddings import imgbeddings
from tqdm import tqdm
import dlib
from imutils import face_utils
import psycopg2
import random
import cv2
import os
import csv

# 1 - Load and Preparing the Dataset

# Data used: [Celebrity Face Image Dataset - Kaggle]
# (https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset)

#local_path = '/content/'
local_path = ''

# 1.1 - Preparing Dataset

# function to generate CSV file from dataset images
def list_image_files(root_dir):
    image_files = []

    # Recursively traverse directories
    for root, dirs, files in os.walk(root_dir):
        # For each file, check if it is an image and add it to the list
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append((os.path.join(root, file), os.path.basename(root)))

    return image_files

# function to get data and time
def datetime_now():
    now = datetime.now()
    # format data and time
    formated_date = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    return formated_date

# Function to create or append to a CSV file
def write_CSV_faces(data_file, goal):
    # Define the path for the CSV file
    if goal == 'data':
        csv_file_path = os.path.join(local_path, 'data', 'data_faces.csv')
        header = ['img_path', 'img_name', 'full_name']
    elif goal == 'detected':
        csv_file_path = os.path.join(local_path, 'data', 'data_faces_detected.csv')
        header = ['date_time', 'img_path', 'img_name', 'full_name']

    # Check if the file already exists
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, 'a', newline='') as csvfile:
        if not file_exists:
            # If the file doesn't exist, write the header
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

        # Write the data
        writer = csv.writer(csvfile)
        writer.writerow(data_file)

# function to extract faces and save information to CSV file
def extract_faces_csv(image_files_list):
    faces = 0

    # array to salve information to use in the future
    out_put_target_file_name = []
    out_put_img_filename = []
    out_put_full_name = []

    j = 0
    # Write the data of each file in the list
    for file_path, full_name in image_files_list:

        # ************************
        # EXTRACT FACES FROM IMAGE
        # ************************
        # loading the image path into file_name variable
        file_name = file_path

        # reading the image
        img = cv2.imread(file_name, 0)

        # creating a black and white version of the image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # creating RGB color
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        face_locations = face_recognition.face_locations(img)

        i = 0
        # for each face detected
        for face_location in face_locations:
            top, right, bottom, left = face_location

            # crop the image to select only the face
            cropped_image = img[top:bottom, left:right]

            # loading the target image path into target_file_name variable
            target_file_name = local_path + 'data/data_faces/' + datetime_now() + "_" + str(j) + '_' + str(i) + '.jpg'

            # Resize the image to 100x100 pixels
            cropped_image_resized = cv2.resize(cropped_image, (100, 100))

            # crop face from image and save
            cv2.imwrite(target_file_name, cropped_image_resized)

            # count faces detected
            faces = faces + 1

            # ************************
            # SAVE CSV FILE
            # ************************
            # write the data of each file in the list
            img_filename = target_file_name.split("/")[-1]
            data_file = ([target_file_name, img_filename, full_name])
            write_CSV_faces(data_file, 'data')

            # put information in array list
            out_put_target_file_name.append(target_file_name)
            out_put_img_filename.append(img_filename)
            out_put_full_name.append(full_name)

            i = i + 1

        j = j + 1

    print(str(faces) + " faces extracted and CSV file saved successfully!")

    return out_put_target_file_name, out_put_img_filename, out_put_full_name

# 1.2 - Load Dataset Images

# function to list sample images from dataset
def generate_random_from_data():
    # Read the CSV file
    df = pd.read_csv(local_path + 'data/data_faces.csv')

    # Select random entries from the DataFrame
    random_rows = df.sample(n=15)

    # Plot the selected images
    plt.figure(figsize=(10, 7))
    for i, (_, row) in enumerate(random_rows.iterrows(), start=1):
        plt.subplot(3, 5, i)
        img = mpimg.imread(row['img_path'])
        plt.imshow(img)
        plt.axis('off')
        plt.title(row['full_name'])

    plt.tight_layout()
    plt.show()

    print('Total images count {}'.format(df.shape[0]))

    # List of corresponding known face names
    known_face_names = df['full_name'].values
    print(list(set(known_face_names)))


# function to commit one register data to PostgreSQL
def commit_to_postgres_one(filename, full_name, img_path):
    try:
        conn = psycopg2.connect("")
        cur = conn.cursor()

        # opening the image from dataset
        img = Image.open(img_path)
        # loading the imgbeddings
        ibed = imgbeddings()
        # caclulating the embeddings
        embedding = ibed.to_embeddings(img)
        cur.execute("INSERT INTO pictures VALUES (%s,%s,%s)", (filename, full_name, embedding[0].tolist()))

        conn.commit()
        cur.close()
        conn.close()
        print("Data committed to PostgreSQL.")
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL:", error)

# function to embedding images from data folders
def findEncodings(df):

    encodeList = []
    # Loop through the DataFrame
    for _, row in df.iterrows():
        image_path = local_path + row['img_path']
        image = cv2.imread(image_path, 0)
        # creating a black and white version of the image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_encode = face_recognition.face_encodings(rgb_image)
        encodeList.append(np.array(image_encode))

    return encodeList

# function to embedding only one image
def embedding_one_face(image):

  # opening the image from dataset
  img = Image.fromarray(image)
  #loading the imgbeddings
  ibed = imgbeddings()
  #caclulating the embeddings
  one_embedding = ibed.to_embeddings(img)[0]

  return one_embedding

# Perform similarity search
"""
To calculate similarity, run the SELECT statements using the built-in vector operators.

As a result, the query computes the L2 distance between the selected vector and the vectors stored in the items table, arrange the results based on the calculated distance, and outputs its top five nearest neighbors (most similar items).

Operators For Calculating Similarity
<-> - Euclidean distance (L2 distance)
<#> - negative inner product
<=> - cosine distance
"""

# function to compare faces using select PostgredSQL
def compare_faces_sql(embedding, operator, number_neighbors, similarity):

  try:
    # connecting to the database
    conn = psycopg2.connect("")
    cur = conn.cursor()

    string_representation = "["+ ",".join(str(x) for x in embedding.tolist()) +"]"

    # This query will select images where the distance between the embedding
    # of the image and some_embedding_value is < 8

    # Euclidean distance (L2 distance)
    if operator == 'euclidean':
      cur.execute("SELECT * FROM pictures WHERE embedding <-> %s < %s ORDER BY embedding <-> %s LIMIT %s;",
                          (string_representation, similarity, string_representation, number_neighbors))

    # negative inner product
    if operator == 'negative':
      cur.execute("SELECT * FROM pictures WHERE embedding <-> %s < %s ORDER BY embedding <#> %s LIMIT %s;",
                          (string_representation, similarity, string_representation, number_neighbors))

    # cosine distance
    if operator == 'cosine':
      cur.execute("SELECT * FROM pictures WHERE embedding <-> %s < %s ORDER BY embedding <=> %s LIMIT %s;",
                          (string_representation, similarity, string_representation, number_neighbors))

    result = cur.fetchall()
    cur.close()
  except psycopg2.Error as e:
      print("Error retrieving similar images:", e)
  finally:
    conn.close()

  return result

# 2 - Detecting faces

# function to verify is file exist in folder
def save_image_faces_detected(img_face_detected, original_image):

    # Original file name
    file_name = os.path.basename(img_face_detected)
    file_path = os.path.dirname(img_face_detected)
    file_path_complete = img_face_detected

    # Check if the file exists
    if os.path.exists(img_face_detected):

        # change the name the file
        # Extract initial name from the file
        extract_initial_name = file_name.split('_')[0]

        # Extract the number from the file
        new_number = int(file_name.split('_')[1].split('.')[0]) + 1

        # Generate the new file name
        file_name = f'{extract_initial_name}_{str(new_number)}.jpg'

        # Save new file name
        file_path_complete =file_path + "/" + file_name
        cv2.imwrite(file_path_complete, original_image)
    else:
        cv2.imwrite(img_face_detected, original_image)
        #print("The file does not exist. Saved with same name")

    return file_path_complete, file_name

# function to plot stamp 'suspect' in frame
def action(action_image, frame_width, frame_height, frame):

    # Read the image to overlay
    image = cv2.imread(action_image, cv2.IMREAD_UNCHANGED)

    # Calculate position to place the image in the middle and bottom
    image_width = image.shape[1]
    image_height = image.shape[0]
    bottom_margin = 20  # Adjust this value as needed

    # Calculate the position to place the image
    x_offset = int((frame_width - image_width) / 2)
    y_offset = frame_height - image_height - bottom_margin

    # Overlay the image onto the video frame
    for c in range(0, 3):
        frame[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] = \
            frame[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] * \
            (1 - image[:, :, 3] / 255.0) + \
            image[:, :, c] * (image[:, :, 3] / 255.0)

    return frame

# function to plot stamp 'suspect' in image
def input_watermark(main_image, overlay):
    # Load the watermark image
    watermark = cv2.imread(overlay, cv2.IMREAD_UNCHANGED)

    # Get dimensions of the main image
    main_height, main_width, _ = main_image.shape

    # Get dimensions of the watermark
    watermark_height, watermark_width, _ = watermark.shape

    # Calculate coordinates to overlay the watermark in the middle of the image
    position_x = (main_width - watermark_width) // 2
    position_y = main_height - watermark_height - 20  # 20 pixels from the bottom edge

    # Overlay the watermark on the main image
    for y in range(watermark_height):
        for x in range(watermark_width):
            # Get RGBA values of the watermark
            b, g, r, alpha = watermark[y, x]
            if alpha > 0:  # Check if the watermark pixel is transparent
                # Overlay the watermark pixel on the main image
                main_image[position_y + y, position_x + x] = [b, g, r]

    return main_image

# 2.1 - Detecting Suspects in Images

def detect_people_faces(image_path):
    # Custom images for suspect detected
    stamp_suspect = 'images/suspect.png'

    faces = 0
    name_detected = []
    # Load an image
    image = cv2.imread(image_path, 0)
    # creating a black and white version of the image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # creating RGB color
    # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    face_locations = face_recognition.face_locations(rgb_image)

    # to get copy original color image
    image_output = cv2.imread(image_path)

    # Iterate through detected faces
    for face_location in face_locations:

        # ************************
        # EXTRACT FACES FROM IMAGE
        # ************************

        # Extract the face encoding from the image
        top, right, bottom, left = face_location
        face_image = rgb_image[top:bottom, left:right]

        # Resize the image to 100x100 pixels
        cropped_image_resized = cv2.resize(face_image, (100, 100))

        # calculating the embeddings
        face_encoding = embedding_one_face(cropped_image_resized)
        # face_encoding = embedding_one_face(face_image)

        # face_encoding = face_recognition.face_encodings(cropped_image_resized)[0]
        # face_encoding = face_recognition.face_encodings(face_image)[0]

        operator = 'euclidean'
        # operator = 'negative'
        # operator = 'cosine'
        number_neighbors = '1'
        # number_neighbors = '5'
        # Compare the detected face encoding with the known face encodings
        # matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        matches = compare_faces_sql(face_encoding, operator, number_neighbors,
                                    9)  # number of similarity = high value, more similar
        # print("Compare faces: " + str(len(matches)))
        # print(matches)

        # If a match is found, use the name of the known face
        face_match = len(matches)
        if face_match > 0:
            # print('Number faces match: ' + str(face_match))
            faces = faces + 1
            # to get full_name from database
            name = matches[0][1]
            # print('Person: ' + name)

            # Draw a box around the face
            cv2.rectangle(image_output, (left, top), (right, bottom), (0, 255, 0), 2)
            # Put the name of the detected face above the box
            cv2.putText(image_output, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # input name inside array
            name_detected.append(name)
            # print(name_detected)

        # else:
        # Draw rectangle around the face
        # cv2.rectangle(image_output, (left, top), (right, bottom), (0, 0, 255), 2)
        # Put the name of the detected face above the box
        # cv2.putText(image_output, "Unknown", (left,top-10), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2)

    if faces > 0:

        # append image 'SUSPECT'
        result_img = input_watermark(image_output, stamp_suspect)

        # ************************
        # SAVE IMAGE FILE
        # ************************
        # Save the image with bounding boxes
        output_image_path = local_path + 'data/faces_detected/from_images/' + datetime_now() + '_IMG_0.jpg'
        output_image_path, img_filename = save_image_faces_detected(output_image_path, result_img)

        # ************************
        # SAVE CSV FILE
        # ************************
        count = 0
        while count < faces:
            count += 1
            for full_name in name_detected:
                # print(data_output)
                data_input_csv = ([datetime_now(), output_image_path, img_filename, full_name])
                write_CSV_faces(data_input_csv, 'detected')
    else:
        # print("No known faces detected")
        output_image_path = ''

    # print(str(faces) + " face(s) detected!")

    return output_image_path

# 2.1.1 - Test Detecting Suspects in Images

# function to load test directory
def load_test_images_directory():
    # Root directory to be predict
    root_directory_test = local_path + 'data/test_detection/images'

    # Get list of image files with directory name
    image_files_predict = list_image_files(root_directory_test)
    # read only first column of matrix
    image_files_predict = [row[0] for row in image_files_predict]

# function to test detection face in image
def test_detection_faces_one_image(find_image):

    load_test_images_directory()
    # try to find faces known
    img = detect_people_faces(find_image)
    if img == '':
        print('No known faces detected')
    elif img != '':
        # Plot the selected images
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.imshow(mpimg.imread(img))
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return img

# 2.2 - Detecting Suspects in Video

# function to find faces in video from database (DVR)
def detect_face_in_video_model(cam, input_image, input_name, input_video):
    # Load the pre-trained face detection and recognition models
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(local_path + "data/models/shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1(
        local_path + "data/models/dlib_face_recognition_resnet_model_v1.dat")

    # Load the reference image
    reference_image = cv2.imread(input_image)
    reference_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    # Detect faces in the reference image
    ref_faces = detector(reference_rgb)
    if len(ref_faces) != 1:
        print("Error: Reference image should contain exactly one face.")
        exit()

    # Get the face embedding of the reference image
    ref_face_shape = shape_predictor(reference_rgb, ref_faces[0])
    ref_face_descriptor = face_rec_model.compute_face_descriptor(reference_rgb, ref_face_shape)

    VIDEO = input_video

    # Define the source: 0 for webcam or the path to your video file
    if cam == True:
        VIDEO = 0

    # Custom images for suspect detected
    suspect = 'images/suspect.png'
    new_image = []

    ######################
    # save image from user
    ######################
    # salve image in data folder
    new_image.append((input_image, input_name))
    # extract face and save information in CSV
    target_file_name, img_filename, full_name = extract_faces_csv(new_image)

    # insert embedding in data PostgreSQL
    commit_to_postgres_one(img_filename[0], full_name[0], input_image)

    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("Error opening original video/camera.")
        return

        # Get the FPS of the original video
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_video_path = local_path + 'data/faces_detected/from_videos/videos/' + datetime_now() + '_VID_0.avi'
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps_original, (frame_width, frame_height))

    skip_frames = 1  # Process every 3rd frame
    frame_count = 0
    # configure threshold to make match
    threshold = 0.7

    def detect_face(frame, person_name):

        act = ""

        # Convert the frame to RGB for face detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faces = detector(frame_rgb)

        # Iterate over each detected face in the frame
        for face in faces:
            # Get the face embedding of the detected face
            face_shape = shape_predictor(frame_rgb, face)
            face_descriptor = face_rec_model.compute_face_descriptor(frame_rgb, face_shape)

            # Compare the face embedding with the reference face embedding
            distance = np.linalg.norm(np.array(ref_face_descriptor) - np.array(face_descriptor))

            # If the distance is below a threshold, consider them the same person
            if distance < threshold:
                # Plot bounding box around the face
                (x, y, w, h) = face_utils.rect_to_bb(face)
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Put the name of the detected face above the box
                cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # append image 'suspect'
                act = suspect

        return frame, act

    if cam == False:
        # Create a tqdm progress bar
        progress_bar = tqdm(total=length, position=0, leave=True)

    while True:

        success, img = cap.read()

        if not success:
            break

        frame_count += 1

        if cam == False:
            # Update the progress bar
            progress_bar.update(1)
            progress_bar.set_description("Processing: %d%%" % (100 * (frame_count) / length), refresh=True)

        # Skip frames to speed up processing
        if frame_count % skip_frames != 0:
            continue
        # img = cv2.resize(img, (resize_width, resize_height))
        result_img, act_action = detect_face(img, input_name)

        if act_action != "":
            # print('Suspect')
            result_img = action(act_action, frame_width, frame_height, result_img)
            # ************************
            # SAVE IMAGE FILE
            # ************************
            # Save the image with bounding boxes
            if cam == True:
                folder = 'from_cams'
                final_name = 'CAM'
            else:
                folder = 'from_video'
                final_name = 'VID'

            date_time = datetime_now()
            target_file_name = local_path + 'data/faces_detected/from_videos/images/' + date_time + '_' + final_name + '_' + str(
                frame_count) + '.jpg'
            cv2.imwrite(target_file_name, result_img)

            # ************************
            # SAVE CSV FILE
            # ************************
            filename = os.path.basename(target_file_name)
            data_input = [date_time, target_file_name, filename, input_name]
            write_CSV_faces(data_input, 'detected')

        # Write the resulting image to the output video file
        # print("Writing frame {} / {}".format(frame_count, length))

        # Write the frame to the output video file
        output_video.write(result_img)

        if cam == True:
            cv2.imshow("Image", result_img)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture and video writer objects
    cap.release()
    output_video.release()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

    if cam == False:
        # Close the progress bar
        progress_bar.close()

    print("Results saved!")

# 2.1 - Plotting video in HTML

# function to show video in HTML
def show_video(path_video):
    from IPython.display import HTML
    from base64 import b64encode

    # Open the video file and read the contents
    with open(path_video, 'rb') as video_file:
        video_content = video_file.read()

    # Encode the video content in base64
    video_base64 = b64encode(video_content).decode()

    # Create the data URL for embedding
    data_url = f"data:video/mp4;base64,{video_base64}"

    # Embed the video in HTML
    html_code = f"""
    <video width=800 controls>
        <source src="{data_url}" type="video/mp4; codecs='avc1.64001E, mp4a.40.2'">
    </video>
    """