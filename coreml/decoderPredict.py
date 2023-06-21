from ctypes.util import find_library
import torch
import ctypes
import numpy as np
from ctypes import cdll, c_double, c_uint, c_int, c_float, c_char_p, c_void_p, POINTER

# input data for trace
modelSize = "tiny"
n_state = { 'tiny': 384, 'base': 512, 'small': 768, 'medium': 1024, 'large': 1280}[modelSize]
n_layer = { 'tiny': 4, 'base': 6, 'small': 12, 'medium': 24, 'large': 32}[modelSize]

dtype1=torch.float32
dtype2=np.float32

bs = 5 # beam_size
text_offset = 10
n_head = 6 # tiny=6, base=8, small=12, medium=16, large=20

# call loadModel
decoderObj = cdll.LoadLibrary('./tiny/decoderWrapper.so')
decoderObj.loadModel.argtypes = [c_char_p, c_int, c_int]
decoderObj.loadModel.restype = c_void_p

mlmodel_handle = decoderObj.loadModel(b'./tiny/CoremlDecoder.mlmodelc', n_layer, n_state)

# call predictWith
decoderObj.predictWith.argtypes = [c_void_p,
                                   POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
                                   c_int, c_int, c_int,
                                   POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float)]
decoderObj.predictWith.restypes = None


# prepare inputs
x = torch.ones((bs, 1, n_state), dtype=dtype1).contiguous()
xPtr = ctypes.cast(x.data_ptr(), POINTER(c_float))

xa = torch.ones((bs, 1500, n_state), dtype=dtype1).contiguous()
xaPtr = ctypes.cast(xa.data_ptr(), POINTER(c_float))

masked_kv_caches = torch.ones((n_layer * 2, bs, text_offset, n_state), dtype=dtype1).contiguous()
mkvPtr = ctypes.cast(masked_kv_caches.data_ptr(), POINTER(c_float))

cross_kv_caches = torch.ones((n_layer * 2, bs, 1500, n_state), dtype=dtype1).contiguous()
ckvPtr = ctypes.cast(cross_kv_caches.data_ptr(), POINTER(c_float))

# prepare outputs
out_x = torch.ones((bs, 1, n_state), dtype=dtype1).contiguous()
outXPtr = ctypes.cast(out_x.data_ptr(), POINTER(c_float))

out_cross_qks = torch.ones((n_layer * bs, n_head, 1, 1500), dtype=dtype1).contiguous()
outCQKPtr = ctypes.cast(out_cross_qks.data_ptr(), POINTER(c_float))

new_masked_kv_caches = torch.ones((n_layer * 2, bs, 1, n_state), dtype=dtype1).contiguous()
outMKVPtr = ctypes.cast(new_masked_kv_caches.data_ptr(), POINTER(c_float))

new_cross_kv_caches = torch.ones(1, dtype=dtype1).contiguous() # this is dummy output
outCKVPtr = ctypes.cast(new_cross_kv_caches.data_ptr(), POINTER(c_float))

# predict
decoderObj.predictWith(mlmodel_handle,
                       xPtr, xaPtr, mkvPtr, ckvPtr,
                       n_layer, n_state, text_offset,
                       outXPtr, outCQKPtr, outMKVPtr, outCKVPtr)

print(out_x[0][0][:2], out_x[bs-1][0][n_state-1])

# call closeModel
decoderObj.closeModel.argtypes = [c_void_p]
decoderObj.closeModel.restypes = None
decoderObj.closeModel(mlmodel_handle)
