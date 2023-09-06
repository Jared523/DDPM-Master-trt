import torch
import os
import torch.nn as nn
class Diff(nn.Module):
    def __init__(self,sd):
        super(Diff, self).__init__()
        self.sd=sd

    def forward(self,x,t):
        t=t.reshape(t.size(-1))
        out=self.sd(x, t)
        return out
    

# import onnx
# outpath=os.path.join("/yzpcode/code/stable-diffusion/out_onnx", 'DDIM.onnx')
# onnx_encoder = onnx.load(outpath)
# # print(onnx_encoder)
# a=10


