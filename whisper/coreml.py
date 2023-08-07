from timeit import default_timer as timer
################################################
# Coreml Encoder part
from ctypes import cdll, c_int, c_float, c_char_p, c_bool, POINTER
import ctypes
import torch

f32Ptr = POINTER(c_float)
logPredictTime = False

totalLoadTime = 0
totalEncoderTime = 0
totalDecoder1Time = 0
totalDecoder256Time = 0
totalCrossKVTime = 0

# some returns are passing by objc in fp16 format
# won't enter python fp32 environment. returns dummy in this case
dummy = torch.ones((1))
class Coreml():
    def __init__(self, n_layer: int, n_state: int, n_head: int, n_vocab: int, modelName):
        self.obj = cdll.LoadLibrary(f'./coreml/{modelName}/coreml.so')
        self.n_layer = n_layer
        self.n_state = n_state
        self.n_head = n_head
        self.n_alignment_head = -1 # for decoder256
        self.bs = -1 # for decoder 1
        self.n_vocab = n_vocab
        self.modelName = modelName
        self.isEncoderLoaded = False
        self.isCrossKVLoaded = False
        self.isDecoder1Loaded = False
        self.isDecoder256Loaded = False

### Encoder #####################################
    def loadEncoder(self):
        global totalLoadTime
        if self.isEncoderLoaded:
            return
        startT = timer()
        self.obj.loadEncoder.argtypes = [c_char_p, c_int, c_int]
        self.obj.loadEncoder.restype = None
        c_string = bytes(f'./coreml/{self.modelName}', 'ascii')
        self.obj.loadEncoder(c_string, self.n_layer, self.n_state)
        self.isEncoderLoaded = True
        totalLoadTime += timer()-startT

    def encoderPredict(self, melSegment):
        global totalEncoderTime
        if not self.isEncoderLoaded:
            self.loadEncoder()
            return
        startT = timer()
        self.obj.encoderPredict.argtypes = [f32Ptr]
        self.obj.encoderPredict.restypes = None

        # force memory continuous, this is very important
        melSegment = melSegment.contiguous()
        melSegmentDataPtr = ctypes.cast(melSegment.data_ptr(), f32Ptr)

        self.obj.encoderPredict(melSegmentDataPtr)
        if logPredictTime:
            print(f"\tcoreml encoder {timer()-startT:.3f}")
        totalEncoderTime += timer() - startT
        return dummy

    def closeEncoder(self):
        if not self.isEncoderLoaded:
            return
        self.obj.closeEncoder.argtypes = None
        self.obj.closeEncoder.restypes = None
        self.obj.closeEncoder()
        self.isEncoderLoaded = False

### CrossKV #####################################
    def loadCrossKV(self):
        global totalLoadTime
        if self.isCrossKVLoaded:
            return
        startT = timer()
        self.obj.loadCrossKV.argtypes = [c_char_p, c_int, c_int]
        self.obj.loadCrossKV.restype = None
        c_string = bytes(f'./coreml/{self.modelName}/CoremlCrossKV.mlmodelc', 'ascii')
        self.obj.loadCrossKV(c_string, self.n_layer, self.n_state)

        n_state = self.n_state
        n_layer = self.n_layer
        n_head = n_state//64

        self.isCrossKVLoaded = True
        totalLoadTime += timer()-startT

    def crossKVPredict(self):
        global totalCrossKVTime
        if not self.isCrossKVLoaded:
            self.loadCrossKV()
            return
        startT = timer()
        self.obj.crossKVPredict.argtypes = None
        self.obj.crossKVPredict.restypes = None

        self.obj.crossKVPredict()

        if logPredictTime:
            print(f"\tcoreml crossKV {timer()-startT:.3f}")
        totalCrossKVTime += timer()-startT
        return dummy, dummy

    def closeCrossKV(self):
        if not self.isCrossKVLoaded:
            return
        self.obj.closeCrossKV.argtypes = None
        self.obj.closeCrossKV.restypes = None
        self.obj.closeCrossKV()
        self.isCrossKVLoaded = False

### Decoder256 #####################################
    def loadDecoder256(self):
        global totalLoadTime
        if self.isDecoder256Loaded:
            return
        startT = timer()

        self.obj.loadDecoder256.argtypes = [c_char_p, c_int, c_int, c_int, c_int, c_int]
        self.obj.loadDecoder256.restype = None
        c_string = bytes(f'./coreml/{self.modelName}/CoremlDecoder256.mlmodelc', 'ascii')
        self.obj.loadDecoder256(c_string, self.n_layer, self.n_state, self.n_head, self.n_alignment_head, self.bs)

        n_head = self.n_head # tiny=6, base=8, small=12, medium=16, large=20
        n_state = self.n_state
        n_layer = self.n_layer
        n_alignment_head = self.n_alignment_head
        max_n_ctx = 256

        dtype1=torch.float32
        # prepare output buffers
        self.out_x256 = torch.ones((1, max_n_ctx, n_state), dtype=dtype1).contiguous()
        self.out_cross_head_weights256 = torch.ones((n_alignment_head, max_n_ctx, 1500), dtype=dtype1).contiguous()
        self.outXPtr256 = ctypes.cast(self.out_x256.data_ptr(), f32Ptr)
        self.outCHWPtr256 = ctypes.cast(self.out_cross_head_weights256.data_ptr(), f32Ptr)
        self.isDecoder256Loaded = True

        totalLoadTime += timer()-startT

    def decoder256Predict(self, x, qk_mask, beam_idx: int):
        global totalDecoder256Time
        if not self.isDecoder256Loaded:
            self.loadDecoder256()
            return
        startT = timer()
        self.obj.decoder256Predict.argtypes = [f32Ptr, f32Ptr,
                                               f32Ptr, f32Ptr, c_int]
        self.obj.decoder256Predict.restypes = None

        # prepare inputs
        x = x.contiguous()
        xPtr = ctypes.cast(x.data_ptr(), f32Ptr)
        qk_mask = qk_mask.contiguous()
        qkMaskPtr = ctypes.cast(qk_mask.data_ptr(), f32Ptr)

        # predict
        self.obj.decoder256Predict(xPtr, qkMaskPtr,
                                   self.outXPtr256, self.outCHWPtr256, beam_idx)
        if logPredictTime:
            print(f"\tcoreml decoder256 {timer()-startT:.3f}")

        totalDecoder256Time += timer()-startT
        return self.out_x256, self.out_cross_head_weights256, dummy

    def closeDecoder256(self):
        if not self.isDecoder256Loaded:
            return
        self.obj.closeDecoder256.argtypes = None
        self.obj.closeDecoder256.restypes = None
        self.obj.closeDecoder256()
        self.isDecoder256Loaded = False

### Decoder1 #####################################
    def loadDecoder1(self):
        global totalLoadTime
        if self.isDecoder1Loaded:
            return
        startT = timer()
        self.obj.loadDecoder1.argtypes = [c_char_p, c_int, c_int, c_int, c_int]
        self.obj.loadDecoder1.restype = None
        c_string = bytes(f'./coreml/{self.modelName}/CoremlDecoder.mlmodelc', 'ascii')
        bs = self.bs # beam_size
        n_head = self.n_head # tiny=6, base=8, small=12, medium=16, large=20
        n_state = self.n_state
        n_layer = self.n_layer
        n_vocab = self.n_vocab
        self.obj.loadDecoder1(c_string, n_layer, n_state, n_head, n_vocab)

        dtype1=torch.float32
        # prepare output buffers
        self.out_x1 = torch.ones((bs, 1, self.n_vocab), dtype=dtype1).contiguous()
        self.new_masked_kv_caches1 = torch.ones((n_layer * 2, bs, 1, n_state), dtype=dtype1).contiguous()
        self.outXPtr1 = ctypes.cast(self.out_x1.data_ptr(), f32Ptr)
        self.isDecoder1Loaded = True
        totalLoadTime += timer()-startT

    def rearrange_mkv(self, indices, text_offset):
        global totalDecoder1Time
        #if logPredictTime:
        #    startT = timer()
        self.obj.rearrange_mkv.argtypes = [POINTER(c_int), c_int]
        self.obj.rearrange_mkv.restypes = None
        indices = indices.to(torch.int32).contiguous()
        indicesPtr = ctypes.cast(indices.data_ptr(), POINTER(c_int))

        # predict
        self.obj.rearrange_mkv(indicesPtr,
                               text_offset)
        #if logPredictTime:
        #    print(f"\tcoreml decoder1 rearrange_mkv {timer()-startT:.3f}")

    def decoder1Predict(self, x, qk_mask, text_offset):
        global totalDecoder1Time
        if not self.isDecoder1Loaded:
            self.loadDecoder1()
            return
        startT = timer()
        self.obj.decoder1Predict.argtypes = [f32Ptr, f32Ptr,
                                             c_int,
                                             f32Ptr]
        self.obj.decoder1Predict.restypes = None

        # prepare inputs
        x = x.contiguous()
        xPtr = ctypes.cast(x.data_ptr(), f32Ptr)
        qk_mask = qk_mask.contiguous()
        qkMaskPtr = ctypes.cast(qk_mask.data_ptr(), f32Ptr)

        # predict
        self.obj.decoder1Predict(xPtr, qkMaskPtr,
                                 text_offset,
                                 self.outXPtr1)
        if logPredictTime:
            print(f"\tcoreml decoder1 {timer()-startT:.3f}")

        totalDecoder1Time += timer() - startT
        return self.out_x1, self.new_masked_kv_caches1

    def closeDecoder1(self):
        if not self.isDecoder1Loaded:
            return
        self.obj.closeDecoder1.argtypes = None
        self.obj.closeDecoder1.restypes = None
        self.obj.closeDecoder1()
        self.isDecoder1Loaded = False


########################################

def showCoremlPredictTime():
    global totalLoadTime
    global totalEncoderTime
    global totalDecoder1Time
    global totalDecoder256Time
    global totalCrossKVTime
    print("--- coreml load -----------------")
    print(f"\ttotal load time    {totalLoadTime:.3f}s")
    print("--- coreml predict --------------")
    print(f"\ttotalEncoder       {totalEncoderTime:.3f}s")
    print(f"\ttotalCrossKV       {totalCrossKVTime:.3f}s")
    print(f"\ttotalDecoder256    {totalDecoder256Time:.3f}s")
    print(f"\ttotalDecoder1      {totalDecoder1Time:.3f}s")
    print(f"\t---")
    print(f"\ttotal predict time {totalEncoderTime+totalCrossKVTime+totalDecoder1Time+totalDecoder256Time:.3f}s")
    print("---------------------------------")

