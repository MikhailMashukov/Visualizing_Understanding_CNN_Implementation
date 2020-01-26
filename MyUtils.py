# from PyQt4.QtCore import *
# from PyQt4.QtGui import *

import psutil
import subprocess
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

def getCpuCoreCount():     # Returns number of cores without considering hyper-threading
    import psutil

    return psutil.cpu_count(False)

def getHardwareStatus(nvidiaSmiExePath):
    # str(psutil.cpu_times()) + '<br>' + \
    status = 'CPU usage: %.1f / %.1f%%, sys: %.1f%%, RAM: %.0f MB, swap: %.0f MB' % \
            (psutil.cpu_percent(interval=0, percpu=False),          # Average from such previous call
             psutil.cpu_percent(interval=0.3, percpu=False),        # Average for 0.3 s
             psutil.cpu_times_percent(interval=0, percpu=False).system,
             psutil.virtual_memory().used / (1 << 20),
             psutil.swap_memory().used / (1 << 20))

    if 0:   # Detailed info like
            # scputimes(user=0.3, nice=0.0, system=0.2, idle=99.3, iowait=0.1, irq=0.0, softirq=0.0, steal=0.0, guest=0.0, guest_nice=0.0)
            # svmem(total=16776122368, available=11851288576, percent=29.4, used=4570382336, free=2955309056, active=7713341440, inactive=5116846080, buffers=1007435776, cached=8242995200, shared=13746176, slab=747556864)
            # sswap(total=38654697472, used=3670016, free=38651027456, percent=0.0, sin=1695744, sout=3063808)
        status += str(psutil.cpu_times_percent(interval=0.3, percpu=False)) + '       ' + \
                str(psutil.cpu_percent(interval=0.3, percpu=False)) + '       ' + \
                str(psutil.cpu_times_percent(interval=0, percpu=False)) + '       ' + \
                str(psutil.cpu_percent(interval=0, percpu=False)) + '       ' + \
                str(psutil.virtual_memory()) + '     ' + \
                str(psutil.swap_memory()) + '     '

    try:
        output = subprocess.check_output([nvidiaSmiExePath],
                                         stderr=subprocess.PIPE)
        for line in output.decode().split('\n'):
            # print(line)
            lineLower = line.lower()
            if lineLower.find('default') >= 0 or lineLower.find('failed') >= 0:
                status += ',\n GPU: %s' % line.strip('\r\n |')
    except Exception as ex:
        print('Exception on nvidia-smi call: %s' % str(ex))
    return status

def getGpuRamFreeMbs(nvidiaSmiExePath):
    try:
        output = subprocess.check_output([nvidiaSmiExePath],
                                         stderr=subprocess.PIPE)
        for line in output.decode().split('\n'):
            # print(line)
            lineLower = line.lower()
            if lineLower.find('default') >= 0 or lineLower.find('failed') >= 0:
                # groups = re.search(r'(\|\s+)(\d+)(m\w+? /\s+)(\d+)(m\w+?\s+\|)', lineLower).groups()
                match = re.search(r'\|\s+(\d+)m\w+? /\s+(\d+)m\w+?\s+\|', lineLower)
                if match:
                    assert len(match.groups()) == 2
                    usedMbs = int(match.groups()[0])
                    totalMbs = int(match.groups()[1])
                    return totalMbs - usedMbs
    except Exception as ex:
        pass

    return 5000


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