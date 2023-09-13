"""
Reads a torch model and then converts it into an onnx model of the same name
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
import torchvision
from easydict import EasyDict
from yaml_config_override import add_arguments

from unet import UNet

if __name__ == "__main__":
    conf = EasyDict(add_arguments())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {conf.model.path}')
    logging.info(f'Using device {device}')
    net = UNet(n_channels=conf.model.n_channels, n_classes=conf.model.n_classes, bilinear=conf.model.bilinear)
    net.to(device=device)
    state_dict = torch.load(conf.model.path, map_location=device)
    state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    net.eval()
    dummy_input = torch.randn(1, 3, 360, 640, device='cuda')
    torch.onnx.export(net, dummy_input, Path(conf.model.path).with_suffix('.onnx'), verbose=True, opset_version=11)
    #check if onnx output close to torch
    onnx_model = onnx.load(Path(conf.model.path).with_suffix('.onnx'))
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(Path(conf.model.path).with_suffix('.onnx'))
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().detach().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    torch_out = net(dummy_input)
    np.testing.assert_allclose(torch_out.cpu().detach().numpy(), ort_outs[0], rtol=1e-03, atol=1e-03)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    import os

    #Convert to TensorRT
    os.system(f"trtexec --onnx={Path(conf.model.path).with_suffix('.onnx')} --saveEngine={Path(conf.model.path).with_suffix('.trt')}")

