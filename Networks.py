import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """VAE Encoder. Output is mu, logVar"""
    def __init__(self, nz, nf, nc=3):
        super(Encoder, self).__init__()
        self.nz = nz
        self.nf = nf
        self.nc = nc
        
        self.conv1 = nn.Conv2d(self.nc, self.nf, 5, 2, 2, bias=False)
        self.conv2 = nn.Conv2d(self.nf, self.nf*2, 5, 2, 2, bias=False)
        self.conv3 = nn.Conv2d(self.nf*2, self.nf*4, 5, 2, 2, bias=False)
        self.fc = nn.Linear(self.nf*4*8*8, 1024, bias=False)
       
        self.bn1 = nn.BatchNorm2d(self.nf, momentum=0.9)
        self.bn2 = nn.BatchNorm2d(self.nf*2, momentum=0.9)
        self.bn3 = nn.BatchNorm2d(self.nf*4, momentum=0.9)
        self.bn4 = nn.BatchNorm1d(1024, momentum=0.9)
  
        self.activ = nn.LeakyReLU(0.2)
        
        self.mu = nn.Linear(1024, self.nz)
        self.logVar = nn.Linear(1024, self.nz)
        
    def forward(self, x):
        x = self.activ(self.bn1(self.conv1(x))) # nf * 32*32
        x = self.activ(self.bn2(self.conv2(x))) # nf*2 * 16*16
        x = self.activ(self.bn3(self.conv3(x))) # nf*4 * 8*8.
        x = x.view(-1, self.nf*4*8*8)
        x = self.activ(self.bn4(self.fc(x)))
        
        return self.mu(x), self.logVar(x)#, out

class Decoder(nn.Module):     
    """Decoder/Generator. Takes in latent vector z and decodes into x_hat"""
    def __init__(self, nz, nf, nc=3):
        super(Decoder, self).__init__()
        self.nz = nz
        self.nf = nf
        self.nc = nc
        
        self.fc = nn.Linear(self.nz, self.nf*4*8*8, bias=False)
        
        self.conv1 = nn.ConvTranspose2d(self.nf*4, self.nf*4, 5, 2, 2, output_padding=1, bias=False)
        self.conv2 = nn.ConvTranspose2d(self.nf*4, self.nf*2, 5, 2, 2, output_padding=1, bias=False)
        self.conv3 = nn.ConvTranspose2d(self.nf*2, self.nf, 5, 2, 2, output_padding=1, bias=False)
        self.conv4 = nn.Conv2d(self.nf, self.nc, 5, 1, 2, bias=False)
       
        self.bn1 = nn.BatchNorm1d(self.nf*4*8*8, momentum=0.9)
        self.bn2 = nn.BatchNorm2d(self.nf*4, momentum=0.9)
        self.bn3 = nn.BatchNorm2d(self.nf*2, momentum=0.9)
        self.bn4 = nn.BatchNorm2d(self.nf, momentum=0.9)
        
        self.activ = nn.LeakyReLU(0.2)
            
    def forward(self, z):
        x = self.activ(self.bn1(self.fc(z)))    # nf*4 * 8*8
        x = x.view(-1, self.nf*4, 8, 8)
        x = self.activ(self.bn2(self.conv1(x))) # nf*4 * 16*16
        x = self.activ(self.bn3(self.conv2(x))) # nf*2 * 32*32
        x = self.activ(self.bn4(self.conv3(x))) # nf * 64*64
        x = self.conv4(x)                       # 3 * 64*64
        return torch.tanh(x)
    
def reparameterize(mu, logVar):
    """Reparameterization takes in the input mu and logVar and sample the mu + std * eps"""
    std = torch.exp(logVar/2)
    eps = torch.randn_like(std)
    return mu + std * eps
    

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder. Returns x_hat, mu, logVar"""
    def __init__(self, nz, nef, ndf, nc=3):
        super(VariationalAutoencoder, self).__init__()
        self.nz = nz # size of hidden code
        self.nef = nef # number of encoder filters
        self.ndf = ndf # number of decoder filters
        self.nc = nc # number of input/output channels
        
        self.Enc = Encoder(self.nz, self.nef, self.nc)
        self.Dec = Decoder(self.nz, self.ndf, self.nc)
        
    def forward(self, x):
        mu, logVar = self.Enc(x)
        z = reparameterize(mu, logVar)
        x_hat = self.Dec(z)
        return x_hat, mu, logVar
    

    
class Discriminator(nn.Module):
    """Discriminator architecture which predicts whether inputs are real or fake."""
    def __init__(self, ndf, nc=3, fc1_out_dim=512):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.fc1_out_dim = fc1_out_dim
        self.register_buffer('buffer', torch.tensor([]))

        self.conv1 = nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(self.ndf*8, 1, 4, 1, 0, bias=False)
        
        self.bn2 = nn.BatchNorm2d(self.ndf*2, momentum=0.9)
        self.bn3 = nn.BatchNorm2d(self.ndf*4, momentum=0.9)
        self.bn4 = nn.BatchNorm2d(self.ndf*8, momentum=0.9)
        self.bn_fc = nn.BatchNorm1d(self.fc1_out_dim)
        
        self.FC1 = nn.Linear(self.ndf*8 * 4*4, self.fc1_out_dim, bias=False)
        self.FC2 = nn.Linear(self.fc1_out_dim, 22*3)
        
        self.activ = nn.LeakyReLU(0.2)
        
    def adversarial(self, x):
        return torch.sigmoid(self.conv5(x))
    
    def xyz(self, x):
        x = x.view(-1, self.ndf*8 * 4*4)
        x = self.activ(self.bn_fc(self.FC1(x)))
        return self.FC2(x)
    
    def forward(self, x, x_hat=None, x_p=None, mode='GAN'):
#         if x_hat == None:
#             x_hat = self.buffer
#         if x_p == None:
#             x_p = self.buffer
#         X = torch.cat((x, x_hat, x_p), 0)       #(batchsize*3, 3, 64, 64)
        X = x
        X = self.activ(self.conv1(X))           # ndf * 32*32
        X = self.activ(self.bn2(self.conv2(X))) # ndf*2 * 16*16
        X = self.activ(self.bn3(self.conv3(X))) # ndf*4 * 8*8
        X = self.activ(self.bn4(self.conv4(X))) # ndf*8 * 4*4
        dis_layer = X # for reconstruction loss
        
        adv_out = self.adversarial(X).squeeze()
        if mode == 'XYZ':
            return adv_out, self.xyz(X), dis_layer
        return adv_out, -1, dis_layer


class PatchDiscriminator(nn.Module):
    """Discriminator returns grid of patch probabilities"""
    def __init__(self, ndf, nc=3, fc1_out_dim=512):
        super(PatchDiscriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.fc1_out_dim=fc1_out_dim

        self.conv1 = nn.Conv2d(self.nc, self.ndf, 4, 2, 1, padding_mode='reflect', bias=False) # 32*32*ndf
        self.conv2 = nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, padding_mode='reflect', bias=False) # 16*16*ndf*2
        self.conv3 = nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, padding_mode='reflect', bias=False) # 8*8*ndf*4
        self.conv4 = nn.Conv2d(self.ndf*4, 1, 4, 1, 1, padding_mode='reflect', bias=False) # 7*7*1
        
        self.bn2 = nn.InstanceNorm2d(self.ndf*2, momentum=0.9)
        self.bn3 = nn.InstanceNorm2d(self.ndf*4, momentum=0.9)
        self.bn4 = nn.BatchNorm2d(self.ndf*4, momentum=0.9)
        self.bn_fc = nn.BatchNorm1d(self.fc1_out_dim, momentum=0.9)
        
        self.FC1 = nn.Linear(self.ndf*4 * 8*8, self.fc1_out_dim, bias=False)
        self.FC2 = nn.Linear(self.fc1_out_dim, 22*3)
        
        self.activ = nn.LeakyReLU(0.2, inplace=True)
    
    def adversarial(self, x): return torch.sigmoid(self.conv4(x))
    
    def xyz(self, x):
        x = self.activ(self.bn4(x))
        x = x.view(-1, self.ndf*4 * 8*8)
        x = self.activ(self.bn_fc(self.FC1(x)))
        return self.FC2(x)
    
    def forward(self, X):
        X = self.activ(self.conv1(X))           # ndf * 32*32
        X = self.activ(self.bn2(self.conv2(X))) # ndf*2 * 16*16
        dis_layer = X # for reconstruction loss
        X = self.activ(self.bn3(self.conv3(X))) # ndf*4 * 8*8
        
        return self.adversarial(X), dis_layer#, self.xyz(X)

#####################################################################################
########## CYCLE GENERATOR ##########################################################
#####################################################################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels), #edit
            nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity() #edit
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class CycleGenerator(nn.Module):
    # https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/CycleGAN/generator_model.py
    def __init__(self, img_channels=3, num_features=64, num_residuals=3):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features), #edit
            nn.LeakyReLU(0.2, inplace=True) #edit
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )
        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
    
    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


    
def weights_init(m):
    """weights initialised with normal dist, 0 mean, 0.02 std.""" 
    classname = m.__class__.__name__
    if classname.find('Conv2') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    #elif classname.find('Linear') != -1:
    #    nn.init.normal_(m.weight.data, 1.0, 0.02)





def perm_gpu_f32(pop_size, num_samples):
    """Use torch.randperm to generate indices on a 32-bit GPU tensor.
    https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/16"""
    return torch.randperm(pop_size, dtype=torch.int32, device='cuda')[:num_samples]

def perm_cpu(pop_size, num_samples):
    """Use torch.randperm to generate indices on a CPU tensor.
    https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/16"""
    return torch.randperm(pop_size)[:num_samples].to('cuda')
        
def noisy_labels(y, p_flip=0.05):
    """https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/"""
    n_select = int(p_flip * y.shape[0]) # num of labels to flip
    flip_idx = perm_cpu( y.shape[0], n_select)
    y[flip_idx] -= 1
    return torch.abs( y )