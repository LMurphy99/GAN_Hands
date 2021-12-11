import torch.nn.functional as F
import torch

Gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).expand(1,3,3,3)
Gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).expand(1,3,3,3)

def R_reg(synth, refined):
    convolution_x = F.conv2d(synth, Gx).detach()
    convolution_y = F.conv2d(synth, Gy).detach()
    convolution_synth = torch.sqrt(convolution_x**2 + convolution_y**2)
    convolution_synth -= convolution_synth.mean(dim=(2,3))[:,None][:,None]
    convolution_synth /= convolution_synth.std(dim=(2,3))[:,None][:,None]
    
    convolution_x = F.conv2d(refined, Gx).detach()
    convolution_y = F.conv2d(refined, Gy).detach()
    convolution_ref = torch.sqrt(convolution_x**2 + convolution_y**2)
    convolution_ref -= convolution_ref.mean(dim=(2,3))[:,None][:,None]
    convolution_ref /= convolution_ref.std(dim=(2,3))[:,None][:,None]
    
    return torch.linalg.norm((convolution_synth - convolution_ref).flatten(), ord=1)
    
def edgeMap(x):
    conv_x = F.conv2d(x, Gx).detach()
    conv_y = F.conv2d(x, Gy).detach()
    conv = torch.sqrt(conv_x**2 + conv_y**2)
    conv -= conv.mean(dim=(2,3))[:,None][:,None]
    conv /= conv.std(dim=(2,3))[:,None][:,None]
    conv = torch.sigmoid(conv)
    numerator = conv - conv.amin(dim=(2,3))[:,None][:,None].permute(0,3,1,2)
    denominator = conv.amax(dim=(2,3))[:,None][:,None].permute(0,3,1,2) - conv.amin(dim=(2,3))[:,None][:,None].permute(0,3,1,2)
    return numerator / denominator