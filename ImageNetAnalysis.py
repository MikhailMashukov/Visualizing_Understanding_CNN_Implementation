# File for running temporary analysis in multiprocess mode

import sys
import traceback
import MultActTops

print('MyApolloDebug.py')
type, value, tb = sys.exc_info()
print(str(traceback.format_tb(tb, limit=6)))
sys.stdout.flush()

if __name__ == '__main__':
    multiOptions = []
    curImageNum = 500

    # Harder tasks
    for curLayerName in ['start_51', 'final_conv_51', 'final_conv_52', 'final_conv_53']:
        multiOptions.append({'curLayerName': curLayerName, 'curImageNum': 500000,
                             'epochNum': 411})

    # First portion of smaller tasks
    for curLayerName in ['start_21']:
        multiOptions.append({'curLayerName': curLayerName, 'curImageNum': 50000,
                             'epochNum': 411})

    # import multiprocessing
    import torch.multiprocessing as multiprocessing

    processes = []
    print('Starting %d processes' % len(multiOptions))
    for options in multiOptions:
        processes.append(multiprocessing.Process(target=MultActTops.workerProcessFunc3_OneOutImage,
                                                 args=(options, )))
    for p in processes:
        p.start()
    for curLayerName in ['final_conv_21', 'final_conv_22']:
        options = {'curLayerName': curLayerName, 'curImageNum': 50000,
                             'epochNum': 411}
        processes[-1].join()
        print('Running another task')
        del processes[-1]

        processes.append(multiprocessing.Process(target=MultActTops.workerProcessFunc3_OneOutImage,
                                                 args=(options, )))
        processes[-1].start()
    print("Awaiting")
    for p in processes:
        p.join()

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
