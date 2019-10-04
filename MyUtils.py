# from PyQt4.QtCore import *
# from PyQt4.QtGui import *

import psutil
import sys
import time

def checkColorComponet(v):
    if v < 0:
        return 0
    elif v > 255:
        return 255
    else:
        return v

def mixColors(color1, coef1, color2, coef2):     # Accepts QColor
    return QColor(checkColorComponet(color1.red()   * coef1 + color2.red()   * coef2),
                  checkColorComponet(color1.green() * coef1 + color2.green() * coef2),
                  checkColorComponet(color1.blue()  * coef1 + color2.blue()  * coef2),
                  checkColorComponet(color1.alpha() * coef1 + color2.alpha() * coef2))

def getPixelFromWidget(widget, point):
    pixmap = QPixmap.grabWidget(widget, QRect(point.x(), point.y(), 1, 1))
    return QColor.fromRgba(pixmap.toImage().pixel(0, 0))

    # image = QImage(1, 1, QImage.Format_RGB32)
    # targetPos = QPoint(0, 0)
    # painter = QPainter(image)
    # sourceArea = QRegion(point.x(), point.y(), 1, 1)
    # widget.render(painter, QRectF(0, 0, 1, 1), QRect(point.x(), point.y(), 1, 1))
    # painter.end()
    # return image.pixel(0, 0)

try:
    import ctypes
    hllDll = ctypes.WinDLL ("User32.dll")

    def getScrollLockState():      # Windows
        # VK_CAPITAL = 0x14
        VK_SCROLL = 0x91
        return hllDll.GetKeyState(VK_SCROLL) & 1
except:
    import os
    import struct
    import fcntl

    def getScrollLockState():      # Linux. Don't work for Basem
        return False

        # DEVICE = '/dev/tty'
        # _KDGETLED = 0x4B31
        # scroll_lock = 0x01
        # # num_lock = 0x02
        # # caps_lock = 0x04
        #
        # fd = os.open(DEVICE, os.O_WRONLY)
        #
        # # ioctl to get state of leds
        # bytes = struct.pack('I', 0)
        # bytes = fcntl.ioctl(fd, _KDGETLED, bytes)    # Returns OSError: [Errno 25] Inappropriate ioctl for device
        # [leds_state] = struct.unpack('I', bytes)
        # return leds_state & scroll_lock


def setProcessPriorityLow():
    p = psutil.Process()
    try:
        p.nice(psutil.IDLE_PRIORITY_CLASS)
        # p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except:
        p.nice(15)     # -20 to 20, bigger number is lower priority


def deep_getsizeof(o, ids):
    d = deep_getsizeof
    if id(o) in ids:
        return 0
    r = sys.getsizeof(o)
    ids.add(id(o))
    if isinstance(o, str) or isinstance(0, bytes):
        return r
    if isinstance(o, dict):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())
    # if isinstance(o, Container):
    try:
        return r + sum(d(x, ids) for x in o)
    except:
        pass
    return r


builtinPrint = print

def print_timeMeasure(*args):
    try:
        t = time.clock()
        builtinPrint("%5.3f (%9.2f) " % (t, (t - print_timeMeasure.prevTime) * 1000000), end='')
        builtinPrint(*args)
        # (%.3f s)%s" % \
        #                   (iterNum, math.sqrt(avgSqDiff), avgSqDiff, time2 - time0, ad
        # time0 = time2
        sys.stdout.flush()
        print_timeMeasure.prevTime = t
    except:
        builtinPrint("seconds from start (passed microseconds)")
        builtinPrint(*args)
        print_timeMeasure.prevTime = time.clock()

print = print_timeMeasure