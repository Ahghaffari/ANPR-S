import configparser
import sys
import os
import cv2
from cryptography.fernet import Fernet
import collections

def decrypt(value):
    f = Fernet('1hK-AOxEVANnJBua2_Z91Cjex5iVheNJEnI9aRUTBtw=')
    decrypted = f.decrypt(value.encode('utf-8'))
    return decrypted.decode()


def encrypt(value):
    f = Fernet('1hK-AOxEVANnJBua2_Z91Cjex5iVheNJEnI9aRUTBtw=')
    encrypted = f.encrypt(value.encode('utf-8')).decode()
    return encrypted


def get_value_encrypted(section, variable_name, type="str"):
    x = Config.get(section, variable_name)
    try:
        value = decrypt(x)
    except:
        value = x
        x = encrypt(x)
        x = x + "****locked****"
        Config.set(section, variable_name, x)
        with open(os.path.abspath(config_file), 'w') as configfile:
            Config.write(configfile)
    if type == "int":
        value = int(value)
    elif type == "float":
        value = float(value)
    return value


# Reading config file
Config = configparser.ConfigParser()
config_file = 'config.ini'
config_file_path = os.path.abspath(config_file)
Config.read(config_file_path)
print("[  CONF  ] Configuration file loaded: ", config_file_path)

MASKF = Config.getboolean('DEPLOY', 'mask')
VERBOSE = Config.getboolean('DEPLOY', 'verbose')

SHOW_RESOLUTION_X = Config.getint('OUTPUT', 'resolution_X')
SHOW_RESOLUTION_Y = Config.getint('OUTPUT', 'resolution_Y')
IMAGE_OUT_PATH = Config.get('OUTPUT', 'IMAGE_OUT_PATH')

CAMERA_BRAND = Config.get('CAMERA', 'brand')
CAMERA_IP = get_value_encrypted('CAMERA', 'ip')
CAM_USER = get_value_encrypted('CAMERA', 'user')
CAM_PASSWORD = get_value_encrypted('CAMERA', 'password')
CAMERA_SET_INIT = Config.getboolean('CAMERA', 'camera_set_init')
CAMERA_SET_AUTO = Config.getboolean('CAMERA', 'camera_set_auto')
QUEUE_SIZE = Config.getint('CAMERA', 'camera_auto_q_size')
GAIN_MINMAX = Config.get('CAMERA', 'gain_min_step_max')
SHUTTER_LIST = Config.get('CAMERA', 'shutter_list')
MIN_SHUTTER = Config.get('CAMERA', 'min_shutter')
PLATE_MEAN_LOW_THR = Config.getint('CAMERA', 'plate_mean_low')
PLATE_MEAN_HIGH_THR = Config.getint('CAMERA', 'plate_mean_high')

SAVE_PIC_DB = Config.getboolean('OPTIONS', 'save_pic_db')
SAVE_PIC_DRIVE = Config.getboolean('OPTIONS', 'save_pic_drive')
SAVE_TO_DB = Config.getboolean('OPTIONS', 'save_to_db')
LIVE = Config.getboolean('OPTIONS', 'live')
CAMERA_NUM = Config.getint('OPTIONS', 'camera_num')
NTP_LIST = Config.get('OPTIONS', 'ntp_list')
SYNC_FLAG = Config.getboolean('OPTIONS', 'sync_time')
LOG_ERR = Config.getboolean('OPTIONS', 'log')
ROTATE = Config.getboolean('OPTIONS', 'rotate')
ROTATION_DEGREE = Config.getint('OPTIONS', 'rotation_degree')
WARPING = Config.getboolean('OPTIONS', 'warping')
WARPING_SET = Config.getboolean('OPTIONS', 'warping_set')

if LOG_ERR:
    # save outputs and errors to files
    sys.stdout = open("log.dat", "w")
    sys.stderr = open('err.dat', 'w')

DB_ADDRESS = get_value_encrypted('DATABASE', 'database_server')
DB_USER = get_value_encrypted('DATABASE', 'database_username')
DB_PASSWORD = get_value_encrypted('DATABASE', 'database_password')

SEG_THR_1 = get_value_encrypted('OCR', 'segmentation_th1', 'float')
SEG_THR_2 = get_value_encrypted('OCR', 'segmentation_th2', 'float')
SEG_THR_3 = get_value_encrypted('OCR', 'segmentation_th3', 'float')
SAME_PLATE_CHAR_MAX = get_value_encrypted('OCR', 'SAME_PLATE_CHAR_MAX', 'int')
SAME_PLATE_CHAR_MIN = get_value_encrypted('OCR', 'SAME_PLATE_CHAR_MIN', 'int')

PLATE_MARGIN = get_value_encrypted('PLATE', 'margin', 'int')
SCALEFACTOR = get_value_encrypted('PLATE', 'scaleFactor', 'float')
MINNEIGHBORS = get_value_encrypted('PLATE', 'minNeighbors', 'int')
MINSIZE_X = get_value_encrypted('PLATE', 'MINSIZE_X', 'int')
MINSIZE_Y = get_value_encrypted('PLATE', 'MINSIZE_Y', 'int')

CASCADE_XML = get_value_encrypted('WEIGHTS', 'a_xml')
NUM_MODEL = get_value_encrypted('WEIGHTS', 'num_model')
CHAR_MODEL = get_value_encrypted('WEIGHTS', 'char_model')
TINY_YOLO_CONFIG_FILE = get_value_encrypted('WEIGHTS', 'config_file')
TINY_YOLO_WEIGHT_FILE = get_value_encrypted('WEIGHTS', 'weight_file')

# Declaring parameters and reading weights
CHAR_DIC = ["A", "B", "P", "W", "X", "J", "D", "C", "S", "T", "E", "G", "L", "M", "N", "V", "H", "Y", "%"]
cascade_model = cv2.CascadeClassifier(os.path.abspath(CASCADE_XML))
number_model = cv2.dnn.readNetFromONNX(os.path.abspath(NUM_MODEL))
character_model = cv2.dnn.readNetFromONNX(os.path.abspath(CHAR_MODEL))
yolo_network = cv2.dnn.readNetFromDarknet(os.path.abspath(TINY_YOLO_CONFIG_FILE), os.path.abspath(TINY_YOLO_WEIGHT_FILE))
CAM_AVG_QUEUE = collections.deque(maxlen=QUEUE_SIZE)
CAM_MEAN_CENTER_QUEUE = collections.deque(maxlen=QUEUE_SIZE)
CAM_MEAN_MINUS_QUEUE = collections.deque(maxlen=QUEUE_SIZE)
mouse_poslist = []
mouse_poslist1 = []