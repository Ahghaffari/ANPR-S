import time
from cryptography.fernet import Fernet
import ctypes


CharArr100 = ctypes.c_char * 100
CharArr1000 = ctypes.c_char * 1000


class ActivationResult(ctypes.Structure):
    _fields_ = [
        ("code", CharArr100),
        ("token", CharArr1000),
        ("key", CharArr1000),
        ("started", CharArr100),
        ("expires", CharArr100),
        ("message", CharArr100),
        ("uuid", CharArr100),
        ("did", CharArr100)]


class SerialNumber:
    _library = None

    activation_result = ActivationResult()

    def __init__(self):
        self._library = ctypes.CDLL("Activator.dll")

        self._library.activateOnline.argtypes = [
            ctypes.POINTER(ActivationResult),
            ctypes.c_wchar
        ]
        self._library.activateOnline.restype = None

    def check_activation_status(self, type):
        self._library.activateOnline(self.activation_result, type)

        code = self.activation_result.code.decode("utf-8")
        message = self.activation_result.message.decode("utf-8")
        token = self.activation_result.token
        key = self.activation_result.key.decode("utf-8")
        expires = self.activation_result.expires.decode("utf-8")
        started = self.activation_result.started.decode("utf-8")
        uuid = self.activation_result.uuid.decode("utf-8")
        did = self.activation_result.did.decode("utf-8")

        try:
            f = Fernet(key)
            result = f.decrypt_at_time(token, current_time=int(time.time()), ttl=100).decode("utf-8")
            if result == uuid:
                return True, uuid, did
            return False, uuid, did
        except:
            return False, uuid, did