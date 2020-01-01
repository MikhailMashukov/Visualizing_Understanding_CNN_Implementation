import matplotlib.pyplot as plt
import DeepOptions

DeepOptions.imagesMainFolder = '/home/ImageNetPart'

import tensorflow as tf

# print(tf.__version__)
# tf.enable_eager_execution()
print(tf.__version__)

from VisJupyterNotUtils import *

# def initColabFolders():
#     from google.colab import drive
#     # !set | more
#     drive.mount('/home/gdrive')
#     %cd "/home/gdrive/My Drive/Visualiz_Zeiler"
#     %pwd
#     !df -hm | grep overlay
#     imagesFolder = '/home/ImageNetPart/'
#     !unzip VKIImageNetPart_TrainTestDivided.zip -d $imagesFolder >/dev/null
#     !df -hm

def initCurTrain():
    from IPython.display import Image
    import os
    import tensorflow as tf
    os.getcwd()
    print(tf.__version__)
    # !pwd
    # %matplotlib inline
    #
    # %mkdir -p "Logs2/src2"
    # # %ls .
    # # %rm -r ~/Visualiz_Zeiler/QtLogs/src
    # %cp -f ./* Logs2/src2/
    #
    #


