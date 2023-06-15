from ctypes.util import find_library
import torch
import ctypes
from ctypes import cdll, c_double, c_uint, c_float, c_char_p, c_void_p, POINTER

# call loadModel
encoderObj = cdll.LoadLibrary('./objcWrapper.o')
encoderObj.loadModel.argtypes = [c_char_p]
encoderObj.loadModel.restype = c_void_p

mlmodel_handle = encoderObj.loadModel(b'./CoremlEncoder.mlmodelc')

# call predictWith
encoderObj.predictWith.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float)]
encoderObj.predictWith.restypes = None
melSegment = torch.ones((1, 80, 3000), dtype=torch.float32)
melSegmentDataPtr = ctypes.cast(melSegment.data_ptr(), POINTER(c_float))

# alloc output buffer
n_state = 384; # tiny=384
output_floats = torch.ones((1500, n_state), dtype=torch.float32)
output_floats_ptr = ctypes.cast(output_floats.data_ptr(), POINTER(c_float))
encoderObj.predictWith(mlmodel_handle, melSegmentDataPtr, output_floats_ptr)
print(output_floats[0][0], output_floats[0][1])

# call closeModel
encoderObj.closeModel.argtypes = [c_void_p]
encoderObj.closeModel.restypes = None
encoderObj.closeModel(mlmodel_handle)
