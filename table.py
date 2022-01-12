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