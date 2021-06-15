import streamlit as st
import os
import cv2 as cv
import random
import numpy as np
import pandas as pd
from datetime import datetime
import imutils
from matplotlib import pyplot as plt

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
                                       {"size": 5, "kernel": 'ellipse',
                                           "type": 'dilate', "iters": 10},
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


TITLE_FONT_SIZE = 20
FIG_SIZE = (12, 12)
BLUR_KERNEL = (3, 3)


def describe_data(numpy_array):
    df = pd.DataFrame({"a": numpy_array.flatten()})
    print(df.describe())


def plot_my_image(img,
                  plot_gray=False,
                  is_gray=False,
                  title=''):
    if is_gray == True:
        pass
    elif is_gray == False:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    fig = plt.figure(figsize=FIG_SIZE)  # create a 5 x 5 figure
    ax = fig.add_subplot(111)

    today = datetime.now()
    today.isoformat()
    ax.set_title(title+' '+today.isoformat(), fontsize=TITLE_FONT_SIZE)
    if plot_gray == True:
        ax.imshow(img, interpolation='none', cmap='gray')
    elif plot_gray == False:
        ax.imshow(img, interpolation='none')
    plt.show()


def crop_background_with_mask(img,
                              mask):
    # load background (could be an image too)
    # white bk, same size and type of image
    bk = np.full(img.shape, 255, dtype=np.uint8)
    # bk = cv.rectangle(bk, (0, 0), (int(img.shape[1] / 2), int(img.shape[0] / 2)), 0, -1)  # rectangles
    #bk = cv.rectangle(bk, (int(img.shape[1] / 2), int(img.shape[0] / 2)), (img.shape[1], img.shape[0]), 0, -1)

    # get masked foreground
    fg_masked = cv.bitwise_and(img,
                               img,
                               mask=mask)
    # get masked background, mask must be inverted
    mask = cv.bitwise_not(mask)
    bk_masked = cv.bitwise_and(bk,
                               bk,
                               mask=mask)
    # combine masked foreground and masked background
    final = cv.bitwise_or(fg_masked,
                          bk_masked)
    mask = cv.bitwise_not(mask)  # revert mask to original
    return final


def rotate_image(img, angle):

    rotated = imutils.rotate(img,
                             angle)
    return rotated


def resize_image(img, scale):

    # scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)

    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized


def rotate_image_2(image,
                   angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    scale = 1.0
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))


def remove_background(img_original,
                      plot_images=True,
                      mythreshold=220,
                      myminLineLength=500,
                      mymaxLineGap=100,
                      mythickness=300,
                      is_gray=True,
                      blurthat=True,
                      blur_kernel=BLUR_KERNEL,
                      area_min=200,
                      ):
    if is_gray == True:
        red_channel = img_original
    elif is_gray == False:
        red_channel = img_original[:, :, 2]
        gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

    if blurthat == True:
        blurred = cv.blur(red_channel, ksize=blur_kernel)
    elif blurthat == False:
        blurred = red_channel

    if plot_images == True:
        plot_my_image(blurred, False, False, title='blurred')

    (mu, sigma) = cv.meanStdDev(red_channel)
    edges = cv.Canny(image=blurred,
                     threshold1=int(mu - sigma),
                     threshold2=int(mu + sigma))

    if plot_images == True:
        plot_my_image(edges, False, title='edges')

    edges_without_dots = remove_white_small_dots(edges, area_min=area_min)

    if plot_images == True:
        plot_my_image(edges_without_dots, False, title='edges 2')

    lines = cv.HoughLinesP(edges_without_dots,
                           rho=1,
                           theta=np.pi / 180,
                           threshold=mythreshold,  # 220
                           minLineLength=myminLineLength,  # 500
                           maxLineGap=mymaxLineGap  # 100
                           )

    print(f'Total lines: {len(lines)}')
    baseline = np.zeros(red_channel.shape)
    temp_img = img_original
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(baseline,
                (x1, y1),
                (x2, y2),
                color=255,
                thickness=mythickness)

    baseline = np.uint8(baseline)

    if plot_images == True:
        plot_my_image(baseline, False, title='baseline')

    return blurred, edges, edges_without_dots, baseline


def normalize_image(img):

    red_channel = img_original[:, :, 2]
    normalized = cv.equalizeHist(red_channel)
    return normalized


def remove_white_small_dots(img, area_min=100):
    # Taken from https://stackoverflow.com/a/57285053/10491422
    # convert to binary by thresholding
    ret, binary_map = cv.threshold(img, 127, 255, 0)

    # do connected components processing
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_map,
                                                                        None,
                                                                        None,
                                                                        None,
                                                                        8,
                                                                        cv.CV_32S)

    # get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= area_min:  # keep
            result[labels == i + 1] = 255

    return result


def remove_background_2(img_original, plot_images=False):

    BLUR_KERNEL = (int(max(img_original.shape)/200),
                   int(max(img_original.shape)/200))

    # 1
    normalized = normalize_image(img_original)
    if plot_images:
        plot_my_image(normalized, False, False, title='org')

    # 2
    blurred = cv.blur(normalized, ksize=BLUR_KERNEL)
    if plot_images:
        plot_my_image(blurred, True, True, title='blr')

    # 3
    res, thr = cv.threshold(blurred, 150, 255, cv.THRESH_BINARY)
    if plot_images:
        plot_my_image(thr, True, True, title='thr')

    # 4
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(thr,
                                                                        None,
                                                                        None,
                                                                        None,
                                                                        8,
                                                                        cv.CV_32S)

    areas = stats[1:, cv.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    max_area = int(img_original.shape[0]*img_original.shape[1]/500)

    for i in range(0, nlabels - 1):
        if areas[i] >= max_area:  # keep
            result[labels == i + 1] = 255

    if plot_images:
        plot_my_image(result, True, True, title=f'result[{max_area}]')

    # 5 erode
    kernel = np.ones((30, 30), np.uint8)
    eroded = cv.dilate(result, kernel, iterations=5)
    if plot_images:
        plot_my_image(eroded, True, True, title='eroded')

    # 6 mask
    masked = crop_background_with_mask(img_original,
                                       eroded)
    if plot_images:
        plot_my_image(masked, False, False, title='masked')

    # 7 detector
    # img_cutted_resized=resize_image(masked,30)
    # cv.imwrite('processed.png',img_cutted_resized)

    # 8 mask inverted
    eroded_inv = cv.bitwise_not(eroded)
    masked_inv = crop_background_with_mask(img_original,
                                           eroded_inv)
    if plot_images:
        plot_my_image(masked_inv, False, False, title='masked_inv')

    return masked, masked_inv, eroded, eroded_inv


if __name__ == '__main__':

    clean_files()

    st.title("Detección de vegetación")
    st.text("Parte de Tesis 2")
    st.text("Aplicación web: Liz F., Milagros M.")
    st.text("Versión: 0.2.10")

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

            img_original = cv.imread(
                input_path+'image.jpg', cv.IMREAD_UNCHANGED)
            masked, masked_inv, eroded, eroded_inv = remove_background_2(img_original,
                                                                         plot_images=False)
            cv.imwrite(input_path+'image_processed.jpg', masked)

            PD = PlantDetection(image=input_path+'image_processed.jpg',
                                morph=15,
                                iterations=2,
                                debug=True,
                                HSV_min=[0, 180, 150],
                                HSV_max=[47, 208, 158],
                                # array=[{"size": 3, "kernel": 'ellipse', "type": 'erode',  "iters": 5},
                                #        {"size": 5, "kernel": 'ellipse',"type": 'dilate', "iters": 10},
                                #        {"size": 3, "kernel": 'ellipse', "type": 'erode',  "iters": 5},
                                #        {"size": 5, "kernel": 'ellipse',"type": 'dilate', "iters": 10},
                                #        ]
                                )
            PD.detect_plants()

            st.subheader("Image contours")
            st.image("image_processed_contours.jpg", caption='Image contours',
                     channels="BGR", use_column_width=True)

            # st.subheader("Image morphed")
            # st.image("image_morphed.jpg", caption="Image morphed",
            #          channels="BGR", use_column_width=True)

            st.subheader("Image morphed original")
            st.image("image_processed_morphed_original.jpg", caption="Image morphed original",
                     channels="BGR", use_column_width=True)


            # Merge masks with background
            img=cv.imread('image_processed_marked.jpg', cv.IMREAD_UNCHANGED)
            result=cv.bitwise_and(img,masked_inv)

            st.subheader("Image marked")
            st.image(result, caption='Image marked',
                     channels="BGR", use_column_width=True)
