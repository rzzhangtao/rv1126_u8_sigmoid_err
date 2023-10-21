import platform
import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

if __name__ == '__main__':
    # Default target and device_id
    target = 'rv1126'
    device_id = None

    # Parameters check
    if len(sys.argv) != 3:
        print('Usage: python {} [onnx_model] [dataset.txt]'.format(sys.argv[0]))
        print('Such as: python {} model.onnx dataset.txt'.format(sys.argv[0]))
        exit(-1)

    ONNX_MODEL = sys.argv[1].strip()
    datasets   = sys.argv[2].strip()
    RKNN_MODEL = ONNX_MODEL + ".rknn"

    # Create RKNN object
    rknn = RKNN()

    if not os.path.exists(ONNX_MODEL):
        print('--> not find {}'.format(ONNX_MODEL))
        exit(-1)
    
    # pre-process config
    #reorder_channel RGB->'0 1 2' BGR->'2 1 0'
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform=[target])
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
	
    if ret != 0:
        print('Load {} failed!'.format(ONNX_MODEL))
        rknn.release()
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=datasets)
    # ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build failed!')
        rknn.release()
        exit(ret)
    print('done')
    
    rknn.accuracy_analysis(inputs='single_img.txt', target=target, dump_file_type='npy')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn failed!')
        rknn.release()
        exit(ret)
    print('done')

    rknn.release()

