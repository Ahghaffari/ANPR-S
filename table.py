import pyodbc
from settings import *


def save_into_database(plate_number, conf, cross_line, car_picture_colored, plate_picture):
    dsn = "Driver={SQL Server};Server=" + DB_ADDRESS + ";Database=SpeedControl;Trusted_Connection=no;uid=" + DB_USER + ";pwd=" + DB_PASSWORD + ";"
    conn = pyodbc.connect(dsn)
    try:
        cursor = conn.cursor()
        sql = "INSERT INTO SpeedControl.dbo.tblLogs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, " \
              "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        cursor.execute(sql,
                       (plate_number, str(conf), '', 0, 0, int(cross_line), '', '', 0, '', '', '', '', 0, 0, 0, '', '', '', 0, 0, 0,
                        pyodbc.Binary(car_picture_colored),
                        pyodbc.Binary(car_picture_colored),
                        cv2.imencode('.jpg', car_picture_colored)[1].tobytes(),
                        cv2.imencode('.jpg', plate_picture)[1].tobytes(), 0, 0, 0, 0, 0, '', '',
                        '', '', '', '', '', '', '', ''))
        cursor.commit()
        print("inserted")
    except pyodbc.Error as ex:
        print(ex)


# def get_last_plate_from_database():
#     conn = pyodbc.connect(
#         "Driver={SQL Server};Server=" + DB_ADDRESS + ";Database=SpeedControl;Trusted_Connection=no;uid=" + DB_USER + ";pwd=" + DB_PASSWORD + ";")
#     try:
#         cursor = conn.cursor()
#         cursor.execute("SELECT TOP 1 PLATE_NUMBER, PLATE_NUMBER_A FROM SpeedControl.dbo.tblLogs ORDER BY ID_CODE desc")
#         row = cursor.fetchone()
#         if row is None:
#             return "none", 0
#         else:
#             return str(row[0]), float(row[1])
#
#     except pyodbc.Error as ex:
#         print(ex)
#         print("[  ERROR  ] : database error")
#         return "0", float(0)
#
#     def create_anpr_speed_table():
#         dsn = f"Driver={SQL_SERVER_DRIVER};Server={SQL_SERVER_SERVER};Database={SQL_SERVER_DATABASE};Trusted_Connection=no;uid={SQL_SERVER_USER};pwd={SQL_SERVER_PASSWORD};"
#         conn = pyodbc.connect(dsn)
#         cursor = conn.cursor()
#         query = (f"USE WIMDB")
#         cursor.execute(query)
#         query = (f"CREATE TABLE ANPR_Speed("
#                  f"ID bigint IDENTITY(1,1) PRIMARY KEY NOT NULL,"
#                  f"MotionTime datetime NOT NULL,"
#                  f"NFrame TINYINT NULL,"
#                  f"Speed int NULL,"
#                  f"StartTime time NULL,"
#                  f"EndTime time NULL,"
#                  f"StartP int NULL, "
#                  f"EndP int NULL, "
#                  f"StartX float NULL, "
#                  f"EndX float NULL, "
#                  f"StartImage varchar(300) NULL, "
#                  f"EndImage varchar(300) NULL)")
#         cursor.execute(query)
#         conn.commit()