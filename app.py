import streamlit as st
import os
import cv2 as cv
import random

from plant_detection.PlantDetection import PlantDetection

input_path = 'input/'
output_path = 'output/'

command_png2mp4_contours = 'ffmpeg -framerate 30 -i ' + \
    'output/contours/' + '%1d_contours.jpg -vcodec libx264 output/output_contours.mp4 -y'
command_png2mp4_marked = 'ffmpeg -framerate 30 -i ' + \
    'output/marked/' + '%1d_marked.jpg -vcodec libx264 output/output_marked.mp4 -y'
command_png2mp4_morphed_original = 'ffmpeg -framerate 30 -i ' + \
    'output/morphed_original/' + \
    '%1d_morphed_original.jpg -vcodec libx264 output/output_morphed_original.mp4 -y'


def clean_files():
    """
    Remove all files in specific paths
    """
    print("----------------CLEAN FILES----------------")

    paths_to_remove = ['input',
                       'output',
                       'output/contours',
                       'output/marked',
                       'output/morphed_original',
                       './']

    for path in paths_to_remove:
        for f in os.listdir(path):
            try:
                print(os.path.join(path, f))
                if ('jpg' or 'png' or 'jpeg' or 'bmp' or 'output') in f:
                    os.remove(os.path.join(path, f))
            except Exception as e:
                print(e)


def split_video_by_frame(video_path, input_drop_path):
    """
    This script will split video into frames with opencv
    """
    print("----------------SPLIT VIDEO BY FRAME----------------")
    # Author: https://gist.github.com/keithweaver/70df4922fec74ea87405b83840b45d57

    cap = cv.VideoCapture(video_path)
    currentFrame = 0
    while(True):
        try:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # Saves image of the current frame in jpg file
            print(input_drop_path)
            name = input_drop_path + str(currentFrame) + '.jpg'
            print('Creating...' + name)

            cv.imwrite(name, frame)

            # To stop duplicate images
            currentFrame += 1
        except Exception as e:
            break
            print(e)

    # When everything done, release the capture
    try:
        cap.release()
        cv.destroyAllWindows()
    except Exception as e:
        print(e)

    return True


def process_images_from_path(input_path):
    """
    Resize all images given a path.
    """
    print("----------------PROCESS IMAGES FROM PATH----------------")

    # Resize all images
    for f in os.listdir(input_path):

        if ('jpg' or 'png' or 'jpeg' or 'bmp') in f:
            print(str(input_path+f))

            PD = PlantDetection(image=input_path+f,
                                morph=15,
                                iterations=2,
                                debug=True,
                                HSV_min=[0, 59, 151],
                                HSV_max=[20, 138, 212],
                                array=[{"size": 3, "kernel": 'ellipse', "type": 'erode',  "iters": 5},
                                       {"size": 5, "kernel": 'ellipse',"type": 'dilate', "iters": 10},
                                       ]
                                )
            PD.detect_plants()

            # print(f[:-4]+'_contours.jpg')
            # print(f[:-4]+'_morphed_original.jpg')
            # print(f[:-4]+'_marked.jpg')

            c = cv.imread(f[:-4]+'_contours.jpg')
            mo = cv.imread(f[:-4]+'_morphed_original.jpg')
            ma = cv.imread(f[:-4]+'_marked.jpg')

            # print(output_path+'contours/'+f+'_contours.jpg')
            # print(output_path+'marked/'+f+'_morphed_original.jpg')
            # print(output_path+'morphed_original/'+f+'_marked.jpg')

            cv.imwrite('output/contours/'+f[:-4]+'_contours.jpg', c)
            cv.imwrite('output/marked/'+f[:-4]+'_marked.jpg', mo)
            cv.imwrite('output/morphed_original/' +
                       f[:-4]+'_morphed_original.jpg', ma)


if __name__ == '__main__':

    clean_files()

    st.title("Plant Detection")
    st.text("Parte de tesis2, detección de vegetación.")
    st.text("Aplicación web: Liz F., Milagros M.")
    st.text("Versión: 0.2.1")

    # Upload file
    st.subheader("1. Elige una imagen o video")
    uploaded_file = st.file_uploader("Elige una imagen compatible",
                                     type=['png', 'jpg', 'bmp', 'jpeg', 'mp4'])

    if uploaded_file is not None:  # File > 0 bytes

        file_details = {"FileName": uploaded_file.name,
                        "FileType": uploaded_file.type,
                        "FileSize": uploaded_file.size}
        st.write(file_details)

        #######################
        # VIDEO UPLOADED FILE
        #######################
        if file_details['FileType'] == 'video/mp4':

            with open(input_path+'temporal.mp4', 'wb') as f:
                f.write(uploaded_file.getbuffer())

            split_video_by_frame(input_path+'temporal.mp4', input_path)

            random_filename = random.choice(os.listdir(input_path))

            st.image(input_path+random_filename, caption='Random image',
                     channels="BGR", use_column_width=True)

            # Executing detection
            st.subheader('Executing detectiong based on computer vision... ')

            try:
                process_images_from_path(input_path)
            except Exception as e:
                print(e)

            # JPG -> MP4
            result1 = os.popen(command_png2mp4_contours).read()
            result2 = os.popen(command_png2mp4_marked).read()
            result3 = os.popen(command_png2mp4_morphed_original).read()

            st.text(result1)
            st.text(result2)
            st.text(result3)

            # Display video
            st.subheader("Video output_contours")
            st.video('output/output_contours.mp4')

            # Display video
            st.subheader("Video output_marked")
            st.video('output/output_marked.mp4')

            # Display video
            st.subheader("Video output_morphed_original")
            st.video('output/output_morphed_original.mp4')

        #######################
        # IMAGE UPLOADED FILE
        #######################
        elif (file_details['FileType'] == 'image/png' or
              file_details['FileType'] == 'image/jpg' or
              file_details['FileType'] == 'image/jpeg' or
              file_details['FileType'] == 'image/bmp'):

            with open(input_path+'image.jpg', 'wb') as f:
                f.write(uploaded_file.getbuffer())

            PD = PlantDetection(image=input_path+'image.jpg',
                                morph=15,
                                iterations=2,
                                debug=True,
                                HSV_min=[0, 97, 0],
                                HSV_max=[24, 190, 102],
                                array=[{"size": 3, "kernel": 'ellipse', "type": 'erode',  "iters": 5},
                                       {"size": 5, "kernel": 'ellipse',"type": 'dilate', "iters": 10},
                                       {"size": 3, "kernel": 'ellipse', "type": 'erode',  "iters": 5},
                                       {"size": 5, "kernel": 'ellipse',"type": 'dilate', "iters": 10},
                                       ]
                                )
            PD.detect_plants()

            st.subheader("Image contours")
            st.image("image_contours.jpg", caption='Image contours',
                     channels="BGR", use_column_width=True)

            # st.subheader("Image morphed")
            # st.image("image_morphed.jpg", caption="Image morphed",
            #          channels="BGR", use_column_width=True)

            st.subheader("Image morphed original")
            st.image("image_morphed_original.jpg", caption="Image morphed original",
                     channels="BGR", use_column_width=True)

            st.subheader("Image marked")
            st.image("image_marked.jpg", caption='Image marked',
                     channels="BGR", use_column_width=True)
