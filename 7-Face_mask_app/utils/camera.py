import cv2
import threading
import time
from imutils.video import VideoStream


def get_snapshot(ip_url):
    
    vs = VideoStream(src=ip_url)
    time.sleep(1.0)
    
    frame = vs.read()
    vs.stop()
    
    if frame is not None:
        return frame
    
    return None


def get_live_frame(vs):
    
    frame = vs.read()
    return frame