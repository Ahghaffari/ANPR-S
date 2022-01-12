import socket
import struct
import sys
import time
import datetime
import win32api
import ctypes, sys



def gettime_ntp(addr='time.nist.gov'):
    TIME1970 = 2208988800
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = '\x1b' + 47 * '\0'
    try:
        client.settimeout(5.0)
        client.sendto(bytes(data.encode("utf-8")), (addr, 123))
        data, address = client.recvfrom(1024)
        if data:
            epoch_time = struct.unpack('!12I', data)[10]
            epoch_time -= TIME1970
            return epoch_time
    except socket.timeout:
        return None


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def main(server_list):
    server_list = server_list.split(",")
    for server in server_list:
        epoch_time = gettime_ntp(server)
        if epoch_time is not None:
            utcTime = datetime.datetime.utcfromtimestamp(epoch_time)
            if is_admin():
                win32api.SetSystemTime(utcTime.year, utcTime.month, utcTime.weekday(), utcTime.day, utcTime.hour,
                                       utcTime.minute, utcTime.second, 0)
            else:
                # Re-run the program with admin rights
                ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)

            localTime = datetime.datetime.fromtimestamp(epoch_time)
            print("Time updated to: " + localTime.strftime("%Y-%m-%d %H:%M") + " from " + server)
            break
        else:
            print("Could not find time from " + server)


if __name__ == "__main__":
    server_list = 'ntp.iitb.ac.in,time.nist.gov,time.windows.com,pool.ntp.org,ir.pool.ntp.org,ntp.day.ir'
    main(server_list)

