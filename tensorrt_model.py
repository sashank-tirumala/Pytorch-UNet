from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import tensorrt_bindings
from PIL import Image

from utils.utils import plot_img_and_maskv3 as plot_img_and_mask


class HostDeviceMem(object):
    """
    Helper data structure for managing memory allocation between host and device.
    """

    def __init__(self, host_mem: np.ndarray, device_mem: np.ndarray) -> None:
        self.host = host_mem
        self.device = device_mem

    def __str__(self) -> str:
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)


class TrtModel:
    """
    Wrapper class for TensorRT models
    """

    def __init__(self, engine_path: Path, max_batch_size=1, dtype=np.float32) -> None:
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.max_batch_size = max_batch_size

        # The below code is adapted from the TensorRT samples
        # It handles the memory allocation of the GPU, because TensorRT runs in a separate thread
        # and we need to make sure that the memory is allocated on the same device as the TensorRT thread
        self.cuda_ctx = cuda.Device(0).make_context()  # create context for device 0
        self.cuda_ctx.push()  # push context to current device
        self.engine = self.load_engine(self.runtime, self.engine_path)
        try:
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            data = self.allocate_buffers()
            self.inputs = data["inputs"]
            self.outputs = data["outputs"]
            self.bindings = data["bindings"]
            self.stream = data["stream"]
        except Exception as e:
            raise RuntimeError("fail to allocate CUDA resources") from e
        finally:
            self.cuda_ctx.pop()  # pop context from current device

    @staticmethod
    def load_engine(
        trt_runtime: tensorrt_bindings.tensorrt.Runtime, engine_path: Path
    ) -> tensorrt_bindings.tensorrt.ICudaEngine:
        """
        Load a serialized engine if available, otherwise build a new TensorRT engine and serialize it.
        """
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self) -> Dict[str, List]:
        """
        Allocate host and device buffers for TensorRT inference.
        """
        inputs: List[HostDeviceMem] = []
        outputs: List[HostDeviceMem] = []
        bindings: List[int] = []
        stream = cuda.Stream()
        return_dict = {
            "inputs": inputs,
            "outputs": outputs,
            "bindings": bindings,
            "stream": stream,
        }

        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))
            if str(self.engine.get_tensor_mode(binding)) == "TensorIOMode.OUTPUT":
                outputs.append(HostDeviceMem(host_mem, device_mem))
            elif str(self.engine.get_tensor_mode(binding)) == "TensorIOMode.INPUT":
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                assert False, "Invalid binding"
        return return_dict

    def __call__(self, x: np.ndarray, batch_size=1) -> int:
        """
        Run inference on a batch of images. Currently only supports batch size of 1.
        """
        x = x.astype(self.dtype)

        np.copyto(self.inputs[0].host, x.ravel())
        self.cuda_ctx.push()
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()
        self.cuda_ctx.pop()
        return self.outputs

class UnetTensorrt():
    def __init__(self, trt_model_path, batch_size=1, input_shape=(3, 360, 640), output_shape=(2, 360, 640)):
        self.trt_model = TrtModel(trt_model_path, max_batch_size=batch_size)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
    
    def __call__(self, x):
        x = self.preprocess(x)
        outputs = self.trt_model(x)
        outputs = outputs[0].host.reshape(self.batch_size, *self.output_shape)
        outputs = outputs.argmax(axis=1)
        return outputs
    
    def preprocess(self, pil_img):
        w, h, c= pil_img.shape
        newW, newH = self.input_shape[1], self.input_shape[2]
        img = cv2.resize(pil_img, (newH, newW), interpolation=cv2.INTER_CUBIC)

        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img
    
if __name__ == "__main__":
    model_path = Path("/home/sashank/data/blade-load-segmentation/000-001/checkpoints/checkpoint_epoch100.trt")
    model = UnetTensorrt(model_path)
    img = Image.open("/home/sashank/data/blade-load-segmentation/000-001/rgb/1693960649942_left.jpg")
    img = np.asarray(img)
    mask = model(img)
    img = model.preprocess(img)
    img = img.transpose((1, 2, 0))
    img = img * 255
    img = img.astype(np.uint8)
    plot_img_and_mask(img, mask[0, :, :], returns_img=False)
    print(np.sum(mask == 0))
    model.trt_model.cuda_ctx.pop()
