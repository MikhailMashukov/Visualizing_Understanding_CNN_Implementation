# File for running temporary analysis in multiprocess mode

import sys
import traceback
import MultActTops

print('MyApolloDebug.py')
type, value, tb = sys.exc_info()
print(str(traceback.format_tb(tb, limit=6)))
sys.stdout.flush()


def runProcesses(multiOptions, wait=True):
    processes = []
    print('Starting %d processes' % len(multiOptions))
    for options in multiOptions:
        processes.append(multiprocessing.Process(target=MultActTops.workerProcessFunc3_OneOutImage,
                                                 args=(options, )))
    for p in processes:
        p.start()
    if wait:
        print("Awaiting")
        for p in processes:
            p.join()

if __name__ == '__main__':
    epochNum = 1204
    imageCount = 2000

    multiOptions = []
    conseqMultiOptions = []

    # Harder tasks
    for curLayerName in ['start_41', 'final_conv_41']: # , 'final_conv_42', 'final_conv_43']:
        multiOptions.append({'curLayerName': curLayerName, 'curImageNum': 200000,
                             'epochNum': 400})

    # Smaller tasks which will be run in parallel 1-after-1
    for curLayerName in ['start_31', 'final_conv_31', 'final_conv_32', 'final_conv_33']:
        conseqMultiOptions.append({'curLayerName': curLayerName, 'curImageNum': 50000,
                             'epochNum': 400})

    # import multiprocessing
    import torch.multiprocessing as multiprocessing

    processes = []
    print('Starting %d processes' % len(multiOptions))
    for options in multiOptions + conseqMultiOptions[:1]:
        processes.append(multiprocessing.Process(target=MultActTops.workerProcessFunc3_OneOutImage,
                                                 args=(options, )))
    for p in processes:
        p.start()
    for options in conseqMultiOptions[1:]:
        processes[-1].join()
        print('Running another task')
        del processes[-1]

        processes.append(multiprocessing.Process(target=MultActTops.workerProcessFunc3_OneOutImage,
                                                 args=(options, )))
        processes[-1].start()
    print("Awaiting")
    for p in processes:
        p.join()

    # multiOptions = []
    # imageCount = 50000
    # for curLayerName in ['final_conv_21', 'conv_311', 'conv_312', 'final_conv_31', 'final_conv_32']:
    #     multiOptions.append({'curLayerName': curLayerName, 'curImageNum': imageCount,
    #                          'epochNum': epochNum})
    # runProcesses(multiOptions)

    # I was unable to implement this with Pool or under Jupyter notebook - was getting
    # 'python daemonic processes are not allowed to have children',
    # 'Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method' or
    # no output from prints...
    # pool = multiprocessing.Pool(len(multiOptions))
    # pool.map(MultActTops.workerProcessFunc3_OneOutImage, multiOptions)
    # self.stopEvent.set()
    print("Finished 1")

    # for p in processes:
    #     p.join()
