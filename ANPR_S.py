import tkinter as tk
import numpy as np
import time
from datetime import datetime
from sys import exit
import activation
import getmac
from settings import *
from table import save_into_database
import sync_time


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
            # window.geometry("269x110")
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
                                            'To activate please call FardIran (+98-21-8234).', font=('calibre', 10), anchor='center')

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


def on_mouse(event, x, y, flags, param):
    global mouse_poslist
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_poslist.append((x, y))


def main():
    print("Started Version 4.2.9 : ")
    global IMAGE_OUT_PATH
    global CAR_OUT_PATH
    global PLATE_OUT_PATH
    global mouse_poslist
    global CAMERA_URL
    global NTP_LIST
    global SYNC_FLAG
    if SYNC_FLAG:
        try:
            sync_time.main(NTP_LIST)
        except:
            pass
    check_licence()
    CAMERA_URL = CAMERA_URL.format(CAM_USER, CAM_PASSWORD)
    CAR_OUT_PATH = IMAGE_OUT_PATH
    if not os.path.exists(IMAGE_OUT_PATH):
        os.mkdir(IMAGE_OUT_PATH)

    CAR_OUT_PATH = IMAGE_OUT_PATH + "/car_" + str(CAMERA_NUM)
    if not os.path.exists(CAR_OUT_PATH):
        os.mkdir(CAR_OUT_PATH)

    PLATE_OUT_PATH = IMAGE_OUT_PATH + "/plate_" + str(CAMERA_NUM)
    if not os.path.exists(PLATE_OUT_PATH):
        os.mkdir(PLATE_OUT_PATH)

    last_plate = "99K999_99"
    last_plate_minimum_conf = 0.05
    cap = cv2.VideoCapture(CAMERA_URL)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(CAMERA_URL)
            print("[ ERROR ] : error initial reading from camera!")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        MASK_RESOLUTION_X, MASK_RESOLUTION_Y = np.shape(gray)
        break

    mask = np.ones((MASK_RESOLUTION_X, MASK_RESOLUTION_Y))
    if MASKF:
        while True:
            cv2.namedWindow('masking_' + str(CAMERA_NUM), cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("masking_" + str(CAMERA_NUM), cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback('masking_' + str(CAMERA_NUM), on_mouse)
            cv2.imshow('masking_' + str(CAMERA_NUM), gray)
            for x, y in mouse_poslist:
                cv2.circle(gray, (x, y), 8, 0, 3)
            cv2.waitKey(1)
            posNp = np.array(mouse_poslist)

            if len(posNp) == 4:
                cv2.destroyWindow('masking_' + str(CAMERA_NUM))
                break

        if posNp[0][1] == posNp[1][1]:
            on_x = True
        else:
            on_x = False
        MASK_SLOP = (posNp[0][0] - posNp[1][0]) / (posNp[0][1] - posNp[1][1])
        if np.abs(MASK_SLOP) < 0.0001:
            MASK_SLOP = 0
        MASK_OFFSET = posNp[0][0] - (posNp[0][1] * MASK_SLOP)
        for x in range(MASK_RESOLUTION_X):
            for y in range(MASK_RESOLUTION_Y):
                if on_x:
                    if x < posNp[0][1]:
                        mask[x, y] = 0
                elif MASK_SLOP > 0 and MASK_SLOP * x + MASK_OFFSET < y:
                    mask[x, y] = 0
                elif MASK_SLOP < 0 and MASK_SLOP * x + MASK_OFFSET > y:
                    mask[x, y] = 0

        MASK_SLOP = np.abs((posNp[1][0] - posNp[2][0]) / (posNp[1][1] - posNp[2][1]))
        if MASK_SLOP < 0.0001:
            MASK_SLOP = 0
        MASK_OFFSET = posNp[1][0] - (posNp[1][1] * MASK_SLOP)
        for x in range(MASK_RESOLUTION_X):
            for y in range(MASK_RESOLUTION_Y):
                if MASK_SLOP * x + MASK_OFFSET < y:
                    mask[x, y] = 0

        if posNp[2][1] == posNp[3][1]:
            on_x = True
        else:
            on_x = False
        MASK_SLOP = (posNp[2][0] - posNp[3][0]) / (posNp[2][1] - posNp[3][1])
        if np.abs(MASK_SLOP) < 0.0001:
            MASK_SLOP = 0
        MASK_OFFSET = posNp[2][0] - (posNp[2][1] * MASK_SLOP)
        for x in range(MASK_RESOLUTION_X):
            for y in range(MASK_RESOLUTION_Y):
                if on_x:
                    if x > posNp[2][1]:
                        mask[x, y] = 0
                elif MASK_SLOP > 0 and MASK_SLOP * x + MASK_OFFSET > y:
                    mask[x, y] = 0
                elif MASK_SLOP < 0 and MASK_SLOP * x + MASK_OFFSET < y:
                    mask[x, y] = 0

        MASK_SLOP = np.abs((posNp[3][0] - posNp[0][0]) / (posNp[3][1] - posNp[0][1]))
        if MASK_SLOP < 0.0001:
            MASK_SLOP = 0
        MASK_OFFSET = posNp[3][0] - (posNp[3][1] * MASK_SLOP)
        for x in range(MASK_RESOLUTION_X):
            for y in range(MASK_RESOLUTION_Y):
                if MASK_SLOP * x + MASK_OFFSET > y:
                    mask[x, y] = 0

        gray = np.multiply(gray, mask).astype(np.uint8)
        gray_ROI = gray[min(posNp[0][1], posNp[1][1]):max(posNp[2][1], posNp[3][1]),
                   min(posNp[0][0], posNp[3][0]):max(posNp[1][0], posNp[2][0])]
        if VERBOSE:
            cv2.imshow("masked ROI_" + str(CAMERA_NUM), cv2.resize(gray_ROI, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
            if cv2.waitKey(0):
                cv2.destroyWindow('masked ROI_' + str(CAMERA_NUM))
        np.save('MASK.npy', mask)
        np.save('points.npy', posNp)

    else:
        try:
            mask = np.load('MASK.npy')
            posNp = np.load('points.npy')
            gray = np.multiply(gray, mask).astype(np.uint8)
            gray_ROI = gray[min(posNp[0][1], posNp[1][1]):max(posNp[2][1], posNp[3][1]),
                       min(posNp[0][0], posNp[3][0]):max(posNp[1][0], posNp[2][0])]
            if VERBOSE:
                cv2.imshow("masked ROI_" + str(CAMERA_NUM),
                           cv2.resize(gray_ROI, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
                if cv2.waitKey(0):
                    cv2.destroyWindow('masked ROI_' + str(CAMERA_NUM))

        except:
            print("[ WARNING ] : Not masked yet, working with full frame or start again and mask it!")
            posNp = [[10, 10], [MASK_RESOLUTION_Y - 10, 10], [MASK_RESOLUTION_Y - 10, MASK_RESOLUTION_X - 10],
                     [10, MASK_RESOLUTION_X - 10]]
            gray_ROI = gray[min(posNp[0][1], posNp[1][1]):max(posNp[2][1], posNp[3][1]),
                       min(posNp[0][0], posNp[3][0]):max(posNp[1][0], posNp[2][0])]
            if VERBOSE:
                cv2.imshow("masked ROI_" + str(CAMERA_NUM),
                           cv2.resize(gray_ROI, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
                if cv2.waitKey(0):
                    cv2.destroyWindow('masked ROI_' + str(CAMERA_NUM))

    while True:
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(CAMERA_URL)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_masked = np.multiply(gray, mask).astype(np.uint8)
        gray_masked_ROI = gray_masked[min(posNp[0][1], posNp[1][1]):max(posNp[2][1], posNp[3][1]),
                          min(posNp[0][0], posNp[3][0]):max(posNp[1][0], posNp[2][0])]
        gray_ROI = gray[min(posNp[0][1], posNp[1][1]):max(posNp[2][1], posNp[3][1]),
                   min(posNp[0][0], posNp[3][0]):max(posNp[1][0], posNp[2][0])]
        plates = cascade_model.detectMultiScale(image=gray_masked_ROI, scaleFactor=SCALEFACTOR,
                                                minNeighbors=MINNEIGHBORS,
                                                minSize=(MINSIZE_X, MINSIZE_Y))

        if len(plates) > 0:

            for (x, y, w, h) in plates:

                plate_img = gray_ROI[y - PLATE_MARGIN:y + h + PLATE_MARGIN, x - PLATE_MARGIN:x + w + PLATE_MARGIN]
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
                        break
                except:
                    pass

        if VERBOSE:
            cv2.imshow("masked_" + str(CAMERA_NUM), cv2.resize(gray_masked, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
            cv2.imshow("gray ROI_" + str(CAMERA_NUM), cv2.resize(gray_ROI, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
            cv2.waitKey(1)
        if LIVE:
            cv2.imshow("LIVE_" + str(CAMERA_NUM), cv2.resize(frame, (SHOW_RESOLUTION_X, SHOW_RESOLUTION_Y)))
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
