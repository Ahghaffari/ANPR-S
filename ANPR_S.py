import tkinter as tk

import cv2
import math
import numpy as np
import time
from datetime import datetime
from sys import exit
import activation
import getmac
from settings import *
from table import save_into_database
import sync_time
import requests
from pyflycap2.interface import CameraContext, GUI, Camera


def decrypt_message(encrypted_message):
    key = b'1hK-AOxEVANnJBua2_Z91Cjex5iVheNJEnI9aRUTBtw='
    f = Fernet(key)
    try:
        decrypted_message = f.decrypt(encrypted_message)
    except:
        print('[  ERORR  ] Invalid token!')
        time.sleep(1)
        exit()
    return decrypted_message.decode()


def is_serial_number_valid(key, serial_number):
    if key + "_Amirhossein" == decrypt_message(serial_number.encode()):
        return True
    else:
        return False


def generate_serial_number(sys_id):
    key = '1hK-AOxEVANnJBua2_Z91Cjex5iVheNJEnI9aRUTBtw='.encode('utf-8')
    f = Fernet(key)
    token_encoded = f.encrypt((sys_id + "_Amirhossein").encode('utf-8')).decode()
    with open("info.txt", "w") as text_file:
        print(token_encoded, file=text_file)


def wo_net_activation(is_activated):
    mac_address = getmac.get_mac_address()
    splited = mac_address.split(":")
    key = ''
    for item in splited:
        key += item
    if is_activated:
        generate_serial_number(key)
        return True

    else:
        try:
            f = open("info.txt", "r")
            SERIAL_NUMBER = f.read()
        except:
            return False
        valid = is_serial_number_valid(key, SERIAL_NUMBER)
        return valid


def check_licence():
    print("Date and Time : ", datetime.now())

    def destroy():
        window.destroy()

    try:
        serial = activation.SerialNumber()
        is_activated, uuid, did = serial.check_activation_status(type='S')
        offline_activation = wo_net_activation(is_activated)
        if is_activated is False and offline_activation is False:
            window = tk.Tk()
            window.title("Registration result")
            window.resizable(0, 0)
            window.protocol('WM_DELETE_WINDOW', destroy)
            window.iconbitmap('b.ico')

            if uuid == "":

                reg = tk.Label(window, text='Please check your internet connection.\n '
                                            'To activate please connect to the internet\n '
                                            ' and call FardIran (+98-21-8234).', font=('calibre', 10), anchor='center')

            else:
                reg = tk.Label(window, text='Device is not activated. Device ID:\n ' +
                                            str(uuid) + '\n' +
                                            'To activate please call FardIran (+98-21-8234).', font=('calibre', 10),
                               anchor='center')

            okVar = tk.IntVar()
            btnOK = tk.Button(window, text="OK", pady=5, font=("calibre", 10),
                              bg='lightgreen', command=lambda: okVar.set(1))
            window.tkraise()
            reg.grid(row=1, column=0)
            btnOK.grid(row=2, column=0)
            window.wait_variable(okVar)

            window.destroy()
            exit()
        else:
            print("successfuly activated!")

    except ValueError:
        print("[ ERROR ] checking licence error!")
        sys.exit()


def segment_plate(image, net, threshold):
    try:
        (H, W) = image.shape[:2]
    except:
        return [], []

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 192), swapRB=True, crop=False)
    net.setInput(blob)

    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.3)

    aa = []
    if len(idxs) > 0:

        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            a = [
                int(classIDs[i]),
                confidences[i],
                boxes[i][0],
                boxes[i][1],
                boxes[i][2],
                boxes[i][3],
            ]

            aa.append(a)

    cropped_image = []
    sorted_array = []
    if len(aa) > 0:
        aa = np.array(aa)
        sorted_array = aa[np.argsort(aa[:, 2])]

        for i in range(len(sorted_array)):
            cropped_image.append(
                image[
                int(sorted_array[i, 3])
                - 1: int(sorted_array[i, 3] + sorted_array[i, 5] + 1),
                int(sorted_array[i, 2])
                - 1: int(sorted_array[i, 2] + sorted_array[i, 4] + 1),
                ]
            )

    return cropped_image, sorted_array


def predict_number(im):
    # K.set_session
    im = im / 255.0
    try:
        im = cv2.resize(im, (28, 28))
        image2 = im.reshape(-1, 28, 28, 1)
        number_model.setInput(image2)
        ynew = number_model.forward()
        # ynew = num_model.predict(image2, batch_size=1)
        lab = ynew.argmax()
    except:
        return "", 0
    return lab, ynew[0][lab]


def predict_character(im):
    # K.set_session
    im = im / 255.0
    im = cv2.resize(im, (28, 28))
    image2 = im.reshape(-1, 28, 28, 1)
    character_model.setInput(image2)
    ynew = character_model.forward()
    # ynew = ch_model.predict(image2, batch_size=1)
    lab = ynew.argmax()
    out = CHAR_DIC[lab]
    return out, ynew[0][lab]


def translate_plate(plate):
    cropped_image, sorted_array = segment_plate(plate, net=yolo_network, threshold=SEG_THR_2)
    enough_box = 1

    if len(cropped_image) < 8:
        cropped_image, sorted_array = segment_plate(plate, net=yolo_network, threshold=SEG_THR_1)
        if len(cropped_image) < 8:
            enough_box = 0

    elif len(cropped_image) > 8:
        cropped_image, sorted_array = segment_plate(plate, net=yolo_network, threshold=SEG_THR_3)
        if len(cropped_image) > 8:
            enough_box = 0

    elif len(cropped_image) == 8:
        pass

    if len(cropped_image) != 8:
        return [], []

    plate = []
    confidences = []
    if enough_box == 1:

        digit, conf = predict_number(cropped_image[0])
        plate.append(digit)
        confidences.append(conf)
        digit, conf = predict_number(cropped_image[1])
        plate.append(digit)
        confidences.append(conf)
        digit, conf = predict_character(cropped_image[2])
        plate.append(digit)
        confidences.append(conf)
        digit, conf = predict_number(cropped_image[3])
        plate.append(digit)
        confidences.append(conf)
        digit, conf = predict_number(cropped_image[4])
        plate.append(digit)
        confidences.append(conf)
        digit, conf = predict_number(cropped_image[5])
        plate.append(digit)
        confidences.append(conf)
        plate.append("_")
        confidences.append(1)
        digit, conf = predict_number(cropped_image[6])
        plate.append(digit)
        confidences.append(conf)
        digit, conf = predict_number(cropped_image[7])
        plate.append(digit)
        confidences.append(conf)
    else:
        return [], []

    plat = ''.join(str(x) for x in plate)

    return plat, confidences


def insert_to_database(save_to_db, save_pic_drive, save_pic_db, verbose, camera_num, last_plate,
                       last_plate_minimum_conf, frame, plate_img, final_plate, confidence):
    min_confidences = np.min(confidence)

    if plate_img is None:
        plate_img = frame

    if len(last_plate) != 9:
        print("plate = ", final_plate)
        filename = final_plate + "_" + str(time.time()) + ".jpg"
        if save_pic_drive:
            cv2.imwrite(CAR_OUT_PATH + "/" + filename, frame)
            cv2.imwrite(PLATE_OUT_PATH + "/" + filename, plate_img)
        if save_pic_db and save_to_db:
            save_into_database(final_plate, str(min_confidences), camera_num, frame, plate_img)
        elif save_to_db:
            save_into_database(final_plate, str(min_confidences), camera_num, np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))

        if verbose:
            cv2.imshow("PLATE_F_" + str(camera_num),
                       cv2.resize(plate_img, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
            cv2.waitKey(1)
        last_plate = final_plate
        last_plate_minimum_conf = min_confidences

    else:
        u = zip(final_plate, last_plate)
        diff = []
        for ii, jj in u:
            if ii != jj:
                diff.append(jj)

        if len(diff) > SAME_PLATE_CHAR_MAX:
            print("plate = ", final_plate)
            filename = final_plate + "_" + str(time.time()) + ".jpg"
            if save_pic_drive:
                cv2.imwrite(CAR_OUT_PATH + "/" + filename, frame)
                cv2.imwrite(PLATE_OUT_PATH + "/" + filename, plate_img)
            if save_pic_db and save_to_db:
                save_into_database(final_plate, str(min_confidences), camera_num, frame, plate_img)
            elif save_to_db:
                save_into_database(final_plate, str(min_confidences), camera_num, np.zeros((1, 1, 3)),
                                   np.zeros((1, 1, 3)))
            if verbose:
                cv2.imshow("PLATE_F_" + str(camera_num),
                           cv2.resize(plate_img, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
                cv2.waitKey(1)
            last_plate = final_plate
            last_plate_minimum_conf = min_confidences

        elif (SAME_PLATE_CHAR_MAX >= len(
                diff) >= SAME_PLATE_CHAR_MIN) and min_confidences > last_plate_minimum_conf:
            print("plate = ", final_plate)
            filename = final_plate + "_" + str(time.time()) + ".jpg"
            if save_pic_drive:
                cv2.imwrite(CAR_OUT_PATH + "/" + filename, frame)
                cv2.imwrite(PLATE_OUT_PATH + "/" + filename, plate_img)
            if save_pic_db and save_to_db:
                save_into_database(final_plate, str(min_confidences), camera_num, frame, plate_img)
            elif save_to_db:
                save_into_database(final_plate, str(min_confidences), camera_num, np.zeros((1, 1, 3)),
                                   np.zeros((1, 1, 3)))
            if verbose:
                cv2.imshow("PLATE_F_" + str(camera_num),
                           cv2.resize(plate_img, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
                cv2.waitKey(1)
            last_plate = final_plate
            last_plate_minimum_conf = min_confidences

    return last_plate, last_plate_minimum_conf

def rotation(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg

def on_mouse_warp(event, x, y, flags, param):
    global mouse_poslist1
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_poslist1.append((x, y))

def on_mouse(event, x, y, flags, param):
    global mouse_poslist
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_poslist.append((x, y))


def set_camera_url(brand, ip, user, password):
    if brand.lower() == "acti":
        url = "http://{0}/cgi-bin/encoder?USER={1}&PWD={2}&SNAPSHOT".format(ip, user, password)
    elif brand.lower() == "milesight":
        url = "http://{0}:{1}@{2}:80/ipcam/mjpeg.cgi".format(user, password, ip)
    else:
        print("[  CONF  ] : Other camera models url")
        url = ip
    return url


def set_camera(state, brand, ip=None, user=None, password=None, gain_list=None, shutter_list=None, plate_img=None):
    global gain, shutter, PLATE_MEAN_LOW_THR, PLATE_MEAN_HIGH_THR, MIN_SHUTTER
    if brand.lower() == "acti":
        if state == "init":
            # compression
            req = "http://{0}:80/cgi-bin/encoder?USER={1}&PWD={2}&VIDEO_RESOLUTION={3}&VIDEO_FPS_NUM={4}&" \
                  "VIDEO_ENCODER={5}&VIDEO_H264_QUALITY={6}". \
                format(ip, user, password, "N1920x1080", "30", "H264", "HIGH")
            requests.get(req)

            # image
            req = "http://{0}:80/cgi-bin/encoder?USER={1}&PWD={2}&VIDEO_BRIGHTNESS={3}&VIDEO_CONTRAST={4}&VIDEO_FLIP_MODE={5}&" \
                  "VIDEO_WDR={6}&VIDEO_DIGITAL_NOISE_REDUCTION={7}" \
                .format(ip, user, password, "50", "80", "1", "AUTO,120,0", "ON")
            requests.get(req)

            # day/night
            req = "http://{0}:80/cgi-bin/encoder?USER={1}&PWD={2}&VIDEO_DAYNIGHT_MODE={3}&VIDEO_DN_IRLED={4}&DAY_GAIN_THD={5}" \
                .format(ip, user, password, "AUTO", "1", "10")
            requests.get(req)

            # exposure / white balance
            req = "http://{0}:80/cgi-bin/encoder?USER={1}&PWD={2}&VIDEO_WB_MODE={3}&VIDEO_EXPOSURE_MODE={4}&" \
                  "VIDEO_SHUTTER_MODE={5}" \
                .format(ip, user, password, "AUTO", "AUTO", "AUTO")
            requests.get(req)
            req = "http://{0}:80/cgi-bin/encoder?USER={1}&PWD={2}&VIDEO_EXPOSURE_GAIN={3}&VIDEO_MAX_SHUTTER={4}" \
                .format(ip, user, password, "80", "8")
            requests.get(req)
        else:

            CAM_AVG_QUEUE.append(cv2.mean(plate_img)[0])
            # Z = np.float32(plate_img.reshape((-1)))
            #
            # # Define criteria, number of clusters(K) and apply kmeans()
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            #
            # ret, label, center = cv2.kmeans(Z, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
            #
            # # Convert back into uint8, and make original image
            # center = np.uint8(center)
            #
            # # Add calculations to queue
            # CAM_MEAN_CENTER_QUEUE.append(center[1][0] / 2 + center[0][0] / 2)
            # CAM_MEAN_MINUS_QUEUE.append(np.abs(center[1][0] - center[0][0]))

            mean = np.mean(CAM_AVG_QUEUE)
            # print(mean, gain, shutter)
            # minus = np.mean(CAM_MEAN_MINUS_QUEUE)
            if mean < PLATE_MEAN_LOW_THR:
                if int(shutter) > int(MIN_SHUTTER):
                    # try:
                    shutter = shutter_list[shutter_list.index(shutter) - 1]
                    req = "http://{0}:80/cgi-bin/encoder?USER={1}&PWD={2}&VIDEO_EXPOSURE_GAIN={3}&VIDEO_SHUTTER_SPEED={4}" \
                        .format(ip, user, password, str(gain), str(shutter))
                    requests.get(req)
                    # except:
                    #     print("[  ERROR  ] : Compare your shutter list with camera config")

                elif gain < int(gain_list[2]):
                    gain = gain + int(gain_list[1])
                    req = "http://{0}:80/cgi-bin/encoder?USER={1}&PWD={2}&VIDEO_EXPOSURE_GAIN={3}&VIDEO_SHUTTER_SPEED={4}" \
                        .format(ip, user, password, str(gain), str(shutter))
                    requests.get(req)

            elif mean > PLATE_MEAN_HIGH_THR:
                if gain > int(gain_list[0]):
                    gain = gain - int(gain_list[1])
                    req = "http://{0}:80/cgi-bin/encoder?USER={1}&PWD={2}&VIDEO_EXPOSURE_GAIN={3}&VIDEO_SHUTTER_SPEED={4}" \
                        .format(ip, user, password, str(gain), str(shutter))
                    requests.get(req)

                elif int(shutter) < int(shutter_list[-1]):
                    # try:
                    shutter = shutter_list[shutter_list.index(shutter) + 1]
                    req = "http://{0}:80/cgi-bin/encoder?USER={1}&PWD={2}&VIDEO_EXPOSURE_GAIN={3}&VIDEO_SHUTTER_SPEED={4}" \
                        .format(ip, user, password, str(gain), str(shutter))
                    requests.get(req)
                    # except:
                    #     print("[  ERROR  ] : Compare your shutter list with camera config")


    elif brand.lower() == "milesight":
        pass
    elif brand.lower() == "pointgrey":
        if state == "init":
            gui = GUI()
            gui.show_selection()
        else:
            # CAM_AVG_QUEUE.append(cv2.mean(plate_img)[0])
            # # Z = np.float32(plate_img.reshape((-1)))
            # #
            # # # Define criteria, number of clusters(K) and apply kmeans()
            # # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            # #
            # # ret, label, center = cv2.kmeans(Z, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
            # #
            # # # Convert back into uint8, and make original image
            # # center = np.uint8(center)
            # #
            # # # Add calculations to queue
            # # CAM_MEAN_CENTER_QUEUE.append(center[1][0] / 2 + center[0][0] / 2)
            # # CAM_MEAN_MINUS_QUEUE.append(np.abs(center[1][0] - center[0][0]))
            #
            # mean = np.mean(CAM_AVG_QUEUE)
            # # print(mean, gain, shutter)
            # # minus = np.mean(CAM_MEAN_MINUS_QUEUE)
            # if mean < PLATE_MEAN_LOW_THR:
            #     if int(shutter) > int(MIN_SHUTTER):
            #         shutter = shutter_list[shutter_list.index(shutter) - 1]
            #
            #     elif gain < int(gain_list[2]):
            #         gain = gain + int(gain_list[1])
            #
            #
            # elif mean > PLATE_MEAN_HIGH_THR:
            #     if gain > int(gain_list[0]):
            #         gain = gain - int(gain_list[1])
            #
            #     elif int(shutter) < int(shutter_list[-1]):
            #         shutter = shutter_list[shutter_list.index(shutter) + 1]
            pass
    else:
        pass



def main():
    print("FardIran ANPR System Started version 1.2.0 ;) ")
    global IMAGE_OUT_PATH, CAMERA_SET_AUTO, CAR_OUT_PATH, PLATE_OUT_PATH, mouse_poslist, mouse_poslist1, CAMERA_IP, CAMERA_BRAND, \
        NTP_LIST, SYNC_FLAG, CAMERA_SET_INIT, gain, shutter

    # sync system time
    if SYNC_FLAG:
        try:
            sync_time.main(NTP_LIST)
        except:
            pass

    #check licence
    check_licence()

    # set camera url based on model
    CAMERA_URL = set_camera_url(CAMERA_BRAND, CAMERA_IP, CAM_USER, CAM_PASSWORD)

    # camera setting initialize
    if CAMERA_SET_INIT:
        set_camera(state="init", brand=CAMERA_BRAND, ip=CAMERA_IP, user=CAM_USER, password=CAM_PASSWORD)

    # create folders for output images
    CAR_OUT_PATH = IMAGE_OUT_PATH
    if not os.path.exists(IMAGE_OUT_PATH):
        os.mkdir(IMAGE_OUT_PATH)

    CAR_OUT_PATH = IMAGE_OUT_PATH + "/car_" + str(CAMERA_NUM)
    if not os.path.exists(CAR_OUT_PATH):
        os.mkdir(CAR_OUT_PATH)

    PLATE_OUT_PATH = IMAGE_OUT_PATH + "/plate_" + str(CAMERA_NUM)
    if not os.path.exists(PLATE_OUT_PATH):
        os.mkdir(PLATE_OUT_PATH)

    # initial plate for comparing
    last_plate = "99K999_99"
    last_plate_minimum_conf = 0.05

    # initial read from camera and get a picture for masking or showing masked frame
    if CAMERA_BRAND == "pointgrey":
        cc = CameraContext()
        cc.force_all_ips()
        cc.rescan_bus()
        cam_list = cc.get_gige_cams()
        c = Camera(serial=cam_list[int(CAMERA_IP)])
        c.connect()
        c.start_capture()
    else:
        cap = cv2.VideoCapture(CAMERA_URL)

    while True:
        if CAMERA_BRAND == "pointgrey":
            try:
                try:
                    c.read_next_image()
                except:
                    continue

                image_dict = c.get_current_image()
                image_bytes = image_dict["buffer"]
                frame = np.frombuffer(image_bytes, dtype=np.uint8).reshape((964, 1288, -1))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            except:
                cc = CameraContext()
                cc.rescan_bus()
                cam_list = cc.get_gige_cams()
                c = Camera(serial=cam_list[int(CAMERA_IP)])
                c.connect()
                c.start_capture()
                print("[ ERROR ] : initial reading from camera!")
                continue

        else:
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(CAMERA_URL)
                print("[ ERROR ] : initial reading from camera!")
                continue

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            gray = frame[:, :, 0]

        break

    # masking
    if MASKF:
        while True:
            cv2.namedWindow('masking_' + str(CAMERA_NUM), cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("masking_" + str(CAMERA_NUM), cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback('masking_' + str(CAMERA_NUM), on_mouse)
            cv2.imshow('masking_' + str(CAMERA_NUM), gray)
            for x, y in mouse_poslist:
                cv2.circle(gray, (x, y), 8, 0, 3)
            posNp = np.array(mouse_poslist)
            key = cv2.waitKey(1)
            if key == 27:  # escape key
                cv2.destroyWindow('masking_' + str(CAMERA_NUM))
                break
        posNp = np.array([posNp], dtype=np.int32)
        gray1 = np.copy(gray)
        gray2 = np.copy(gray)
        mask = cv2.fillPoly(gray1, posNp, 0)
        mask = np.logical_not(mask).astype(np.uint8)
        gray = np.multiply(gray, mask).astype(np.uint8)
        gray_ROI = gray2[min(posNp[0, :, 1]):max(posNp[0, :, 1]), min(posNp[0, :, 0]):max(posNp[0, :, 0])]
        if VERBOSE:
            cv2.imshow("masked ROI_" + str(CAMERA_NUM), cv2.resize(gray_ROI, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
            cv2.imshow("masked_" + str(CAMERA_NUM), cv2.resize(gray, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
            if cv2.waitKey(0):
                cv2.destroyWindow('masked ROI_' + str(CAMERA_NUM))
                cv2.destroyWindow('masked_' + str(CAMERA_NUM))
        np.save('points.npy', posNp)

    else:
        try:
            posNp = np.load('points.npy')
            gray1 = np.copy(gray)
            gray2 = np.copy(gray)
            mask = cv2.fillPoly(gray1, posNp, 0)
            mask = np.logical_not(mask).astype(np.uint8)
            gray = np.multiply(gray, mask).astype(np.uint8)
            gray_ROI = gray2[min(posNp[0, :, 1]):max(posNp[0, :, 1]), min(posNp[0, :, 0]):max(posNp[0, :, 0])]
            if VERBOSE:
                cv2.imshow("masked ROI_" + str(CAMERA_NUM),
                           cv2.resize(gray_ROI, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
                cv2.imshow("masked_" + str(CAMERA_NUM), cv2.resize(gray, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
                if cv2.waitKey(0):
                    cv2.destroyWindow('masked ROI_' + str(CAMERA_NUM))
                    cv2.destroyWindow('masked_' + str(CAMERA_NUM))

        except:
            print("[ WARNING ] : Not masked yet, working with full frame or start again and mask it!")
            gray_shape = np.shape(gray)
            posNp = np.array([[(0, 0), (0, gray_shape[0]), (gray_shape[1], gray_shape[0]), (gray_shape[1], 0)]], dtype=np.int32)
            gray_ROI = gray
            if VERBOSE:
                cv2.imshow("masked ROI_" + str(CAMERA_NUM), cv2.resize(gray_ROI, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
                if cv2.waitKey(0):
                    cv2.destroyWindow('masked ROI_' + str(CAMERA_NUM))

    if WARPING_SET:
        while True:
            cv2.namedWindow('warping_' + str(CAMERA_NUM), cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("warping_" + str(CAMERA_NUM), cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback('warping_' + str(CAMERA_NUM), on_mouse_warp)
            cv2.imshow('warping_' + str(CAMERA_NUM), gray_ROI)
            for x, y in mouse_poslist1:
                cv2.circle(gray_ROI, (x, y), 8, 0, 3)
            pts_src = np.array(mouse_poslist1)
            key = cv2.waitKey(1)
            if key == 27:  # escape key
                cv2.destroyWindow('warping_' + str(CAMERA_NUM))
                break

        pts_dst = np.array([[min(pts_src[0, 0], pts_src[2, 0]), min(pts_src[0, 1], pts_src[1, 1])],
                            [max(pts_src[1, 0], pts_src[2, 0]), min(pts_src[0, 1], pts_src[1, 1])],
                            [max(pts_src[1, 0], pts_src[2, 0]), max(pts_src[2, 1], pts_src[3, 1])],
                            [min(pts_src[0, 0], pts_src[2, 0]), max(pts_src[2, 1], pts_src[3, 1])]])
        try:
            warp_mat, status = cv2.findHomography(pts_src, pts_dst)
            np.save("warp.npy", warp_mat)
        except:
            print("[ ERROR  ] : Error in selecting points for warping!")

    elif WARPING:
        try:
            warp_mat = np.load("warp.npy")
        except:
            print("[ WARNING  ] : warping mask couldn`t find. please warp it first then you can use it.")
            while True:
                cv2.namedWindow('warping_' + str(CAMERA_NUM), cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("warping_" + str(CAMERA_NUM), cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.setMouseCallback('warping_' + str(CAMERA_NUM), on_mouse_warp)
                cv2.imshow('warping_' + str(CAMERA_NUM), gray_ROI)
                for x, y in mouse_poslist1:
                    cv2.circle(gray_ROI, (x, y), 8, 0, 3)
                pts_src = np.array(mouse_poslist1)
                key = cv2.waitKey(1)
                if key == 27:  # escape key
                    cv2.destroyWindow('warping_' + str(CAMERA_NUM))
                    break

            pts_dst = np.array([[min(pts_src[0, 0], pts_src[2, 0]), min(pts_src[0, 1], pts_src[1, 1])],
                                [max(pts_src[1, 0], pts_src[2, 0]), min(pts_src[0, 1], pts_src[1, 1])],
                                [max(pts_src[1, 0], pts_src[2, 0]), max(pts_src[2, 1], pts_src[3, 1])],
                                [min(pts_src[0, 0], pts_src[2, 0]), max(pts_src[2, 1], pts_src[3, 1])]])
            try:
                warp_mat, status = cv2.findHomography(pts_src, pts_dst)
                np.save("warp.npy", warp_mat)
            except:
                print("[ ERROR  ] : Error in selecting points for warping!")


    # initial manual camera setting for manual gain and shutter setting
    if not CAMERA_SET_AUTO:
        if CAMERA_BRAND.lower() == "acti":
            # set gain and shutter manual
            req = "http://{0}:80/cgi-bin/encoder?USER={1}&PWD={2}&VIDEO_EXPOSURE_MODE={3}" \
                .format(CAMERA_IP, CAM_USER, CAM_PASSWORD, "Manual")
            requests.get(req)
            req = "http://{0}:80/cgi-bin/encoder?USER={1}&PWD={2}&VIDEO_EXPOSURE_GAIN&VIDEO_SHUTTER_SPEED" \
                .format(CAMERA_IP, CAM_USER, CAM_PASSWORD)
            requests.get(req)
            res = requests.get(req).content.decode("utf-8").split("'")
            gain = int(res[1])
            shutter = str(res[3])
            gain_list = GAIN_MINMAX.split(",")
            shutter_list = SHUTTER_LIST.split(",")

            if int(shutter) < int(MIN_SHUTTER):
                shutter = MIN_SHUTTER

            if int(gain) > int(gain_list[2]):
                gain = int(gain_list[2])
            elif int(gain) < int(gain_list[0]):
                gain = int(gain_list[0])

    while True:

        # read from camera
        if CAMERA_BRAND.lower() == "pointgrey":
            try:
                try:
                    c.read_next_image()
                except:
                    continue
                image_dict = c.get_current_image()
                image_bytes = image_dict["buffer"]
                frame = np.frombuffer(image_bytes, dtype=np.uint8).reshape((964, 1288, -1))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            except:
                cc = CameraContext()
                cc.rescan_bus()
                cam_list = cc.get_gige_cams()
                c = Camera(serial=cam_list[int(CAMERA_IP)])
                c.connect()
                c.start_capture()
                print("[ ERROR ] : initial reading from camera!")
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(CAMERA_URL)
                print("[ ERROR ] : initial reading from camera!")
                continue

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:

            gray = frame[:, :, 0]

        # process taken frame
        gray1 = np.copy(gray)
        gray2 = np.copy(gray)
        mask = cv2.fillPoly(gray1, posNp, 0)
        mask = np.logical_not(mask).astype(np.uint8)
        gray = np.multiply(gray, mask).astype(np.uint8)
        gray_ROI = gray2[min(posNp[0, :, 1]):max(posNp[0, :, 1]), min(posNp[0, :, 0]):max(posNp[0, :, 0])]
        gray_masked = np.multiply(gray, mask).astype(np.uint8)
        gray_masked_ROI = gray_masked[min(posNp[0, :, 1]):max(posNp[0, :, 1]), min(posNp[0, :, 0]):max(posNp[0, :, 0])]
        if WARPING:
            gray_masked_ROI = cv2.warpPerspective(gray_masked_ROI, warp_mat, (gray_masked_ROI.shape[1], gray_masked_ROI.shape[0]))
            gray_ROI = cv2.warpPerspective(gray_ROI, warp_mat,
                                                  (gray_masked_ROI.shape[1], gray_masked_ROI.shape[0]))
        if ROTATE:
            gray_masked_ROI = rotation(gray_masked_ROI, ROTATION_DEGREE)
            gray_ROI = rotation(gray_ROI, ROTATION_DEGREE)
        plates = cascade_model.detectMultiScale(image=gray_masked_ROI, scaleFactor=SCALEFACTOR,
                                                minNeighbors=MINNEIGHBORS,
                                                minSize=(MINSIZE_X, MINSIZE_Y))

        if len(plates) > 0:

            for (x, y, w, h) in plates:
                gray_ROI_shape = np.shape(gray_ROI)
                plate_img = gray_ROI[max(y - PLATE_MARGIN, 1):min(y + h + PLATE_MARGIN, gray_ROI_shape[0]-1), max(x - PLATE_MARGIN, 1):min(x + w + PLATE_MARGIN, gray_ROI_shape[1]-1)]

                if VERBOSE:
                    cv2.imshow("PLATE_R_" + str(CAMERA_NUM),
                               cv2.resize(plate_img, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
                    cv2.waitKey(1)

                try:
                    final_plate, confidence = translate_plate(plate_img)
                    if len(str(final_plate)) == 9:
                        last_plate, last_plate_minimum_conf = insert_to_database(verbose=VERBOSE,
                                                                                 save_pic_db=SAVE_PIC_DB,
                                                                                 save_pic_drive=SAVE_PIC_DRIVE,
                                                                                 save_to_db=SAVE_TO_DB,
                                                                                 camera_num=CAMERA_NUM,
                                                                                 last_plate=last_plate,
                                                                                 last_plate_minimum_conf=last_plate_minimum_conf,
                                                                                 frame=frame,
                                                                                 plate_img=plate_img,
                                                                                 final_plate=final_plate,
                                                                                 confidence=confidence)
                        if VERBOSE:
                            print("[  INFO  ] OCR successful. Translated plate: ", final_plate)
                            cv2.imshow("PLATE_O_" + str(CAMERA_NUM),
                                       cv2.resize(plate_img, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
                            cv2.waitKey(1)
                        if not CAMERA_SET_AUTO:
                            set_camera(state="stream", brand=CAMERA_BRAND, ip=CAMERA_IP, user=CAM_USER, password=CAM_PASSWORD,
                                       gain_list=gain_list, shutter_list=shutter_list, plate_img=plate_img)
                        break

                except:
                    pass

        if VERBOSE:
            cv2.imshow("masked_" + str(CAMERA_NUM), cv2.resize(gray_masked, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
            cv2.imshow("gray ROI_" + str(CAMERA_NUM), cv2.resize(gray_ROI, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
            cv2.imshow("gray_masked_ROI_" + str(CAMERA_NUM), cv2.resize(gray_masked_ROI, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
            cv2.waitKey(1)
        if LIVE:
            cv2.imshow("LIVE_" + str(CAMERA_NUM), cv2.resize(frame, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
