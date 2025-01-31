# ===========================
# Cell 1: Environment & ZIP Extraction
# ===========================

import os
import zipfile
import torch

# (A) Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# (B) Paths for ZIP and extraction
zip_file_path = '//kaggle/input/pretrainer-lndq/_output_.zip'  # Adjust if needed
extract_folder = '/kaggle/working/'
os.makedirs(extract_folder, exist_ok=True)

# (C) Extract the zip containing pretrained models
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# (D) List extracted files/folders
extracted_files = os.listdir(extract_folder)
print("Extracted files/folders:", extracted_files)

# (E) Define the path to your pretrained_models directory
pretrained_models_dir = os.path.join(extract_folder, 'pretrained_models')
print("Pretrained models directory:", pretrained_models_dir)

# If you have additional environment setup or library imports, you can include them here.
print("\n[Cell 1 Complete] - ZIP extracted, device set, environment ready.\n")


# ============================
# Cell 2: Dataset & Architecture Definitions
# ============================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# For CIFAR ResNet
from torchvision.models.resnet import BasicBlock

# For other architectures
import torchvision.models as tv_models

####################################################
# 1) DATA LOADING - CIFAR-10
####################################################

# Paths: we assume you have:
#    /kaggle/working/data/cifar-10-batches-py/ (CIFAR-10)
#    /kaggle/working/data/tiny-imagenet-200/   (TinyImageNet)
#
# If needed, adjust the 'root' argument below to match your folder structure exactly.

# A) CIFAR-10 Transforms
transform_cifar_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247,   0.243,   0.261))
])

# Often for test we do not do random flips/crops, but we can keep consistent if you want.
transform_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247,   0.243,   0.261))
])

# B) oad CIFAR-10 from /kaggle/working/data
cifar_data_path = "/kaggle/working/data/"
trainset_cifar = torchvision.datasets.CIFAR10(
    root=cifar_data_path,
    train=True,
    download=False,  # set True if not downloaded
    transform=transform_cifar_train
)
trainloader_cifar = DataLoader(trainset_cifar, batch_size=128, shuffle=True, num_workers=2)

testset_cifar = torchvision.datasets.CIFAR10(
    root=cifar_data_path,
    train=False,
    download=False,  # set True if not downloaded
    transform=transform_cifar_test
)
testloader_cifar = DataLoader(testset_cifar, batch_size=100, shuffle=False, num_workers=2)

print(f"✅ CIFAR-10 train size: {len(trainset_cifar)}, test size: {len(testset_cifar)}")

####################################################
# 2) DATA LOADING - TinyImageNet
####################################################

# A) TinyImageNet transforms (similar to ImageNet style)
transform_imagenet_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

transform_imagenet_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

tiny_data_path = "/kaggle/working/data/tiny-imagenet-200"
if os.path.exists(tiny_data_path):
    # training set
    trainset_imagenet = datasets.ImageFolder(
        root=os.path.join(tiny_data_path, "train"),
        transform=transform_imagenet_train
    )
    trainloader_imagenet = DataLoader(trainset_imagenet, batch_size=128, shuffle=True, num_workers=2)

    # validation set
    testset_imagenet = datasets.ImageFolder(
        root=os.path.join(tiny_data_path, "val"),
        transform=transform_imagenet_test
    )
    testloader_imagenet = DataLoader(testset_imagenet, batch_size=100, shuffle=False, num_workers=2)

    print(f"✅ TinyImageNet train size: {len(trainset_imagenet)}, val size: {len(testset_imagenet)}")
else:
    trainloader_imagenet = None
    testloader_imagenet  = None
    print("❌ TinyImageNet not found at /kaggle/working/data/tiny-imagenet-200. Skipping.")

####################################################
# 3) Define Custom ResNet for CIFAR (ResNet_CIFAR)
####################################################
class ResNet_CIFAR(nn.Module):
    """
    A small ResNet model for CIFAR-10, using BasicBlock from torchvision.
    Example usage: 
      resnet20 = ResNet_CIFAR([3, 3, 3], num_classes=10)
      resnet32 = ResNet_CIFAR([5, 5, 5], num_classes=10)
      resnet56 = ResNet_CIFAR([9, 9, 9], num_classes=10)
    """
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(BasicBlock, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers  = []
        for st in strides:
            downsample = None
            if st != 1 or self.in_planes != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=st, bias=False),
                    nn.BatchNorm2d(planes)
                )
            layers.append(block(self.in_planes, planes, st, downsample))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Quick instantiation helpers
def resnet20_cifar(num_classes=10):
    return ResNet_CIFAR([3, 3, 3], num_classes)

def resnet32_cifar(num_classes=10):
    return ResNet_CIFAR([5, 5, 5], num_classes)

def resnet56_cifar(num_classes=10):
    return ResNet_CIFAR([9, 9, 9], num_classes)

####################################################
# 4) Define VGG16 & WRN for CIFAR
####################################################
def vgg16_cifar(num_classes=10):
    """
    Loads torchvision's VGG16, modifies the last FC to have num_classes outputs.
    """
    model = tv_models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def wrn50_cifar(num_classes=10):
    """
    Loads wide_resnet50_2, modifies final fc for CIFAR-10.
    """
    model = tv_models.wide_resnet50_2(pretrained=False)
    model.fc = nn.Linear(2048, num_classes)
    return model

####################################################
# 5) Load Pretrained ImageNet Models (Optional)
####################################################
# If you want ResNet18, ResNet34, ResNet50, AlexNet from torchvision, with ImageNet pretrained:

def load_imagenet_resnet18():
    net = tv_models.resnet18(pretrained=True)
    return net

def load_imagenet_resnet34():
    net = tv_models.resnet34(pretrained=True)
    return net

def load_imagenet_resnet50():
    net = tv_models.resnet50(pretrained=True)
    return net

def load_imagenet_alexnet():
    net = tv_models.alexnet(pretrained=True)
    return net

print("\n[Cell 2 Complete] - Datasets (CIFAR-10, TinyImageNet) and model architectures are now defined.")


# ============================
# Cell 3: LDNQ + Other Quantization Methods (From Paper's Source)
# ============================

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

##############################
# 1) Low-Level Utilities
##############################

def get_error(theta_B, hessian, theta_0):
    """
    Calculate delta^T * H * delta  (Second-order error estimate).
    delta = theta_B - theta_0
    If you do not have a Hessian, you can pass an identity or skip using this.
    """
    delta = theta_B - theta_0
    error = np.trace(np.dot(np.dot(delta.T, hessian), delta))
    return error

def unfold_kernel(kernel):
    """
    In PyTorch: kernel shape = [out_channel, in_channel, height, width].
    Unfold into 2D: [in_channel*height*width, out_channel].
    """
    k_shape = kernel.shape
    weight = np.zeros([k_shape[1]*k_shape[2]*k_shape[3], k_shape[0]])
    for i in range(k_shape[0]):
        weight[:, i] = np.reshape(kernel[i, :, :, :], [-1])
    return weight

def fold_weights(weights, kernel_shape):
    """
    Fold 2D array back to 4D kernel shape [out_channel, in_channel, h, w].
    """
    kernel = np.zeros(shape=kernel_shape)
    for i in range(kernel_shape[0]):
        kernel[i,:,:,:] = weights[:, i].reshape([kernel_shape[1], kernel_shape[2], kernel_shape[3]])
    return kernel


##############################
# 2) Discrete Quantization Routines
##############################

def quantize(V, alpha, beta=0, kbits=3):
    """
    Given a real matrix V, quantize into sets like {-1,0,+1}, etc. 
    Currently supports 3,5,7,9,11-level schemes as in the paper’s code.
    """
    if kbits == 3:
        # Ternary
        pos = 1.0 * (V > (alpha/2.0))
        neg = -1.0 * (V < -(alpha/2.0))
        Q = pos + neg

    elif kbits == 5:
        # Levels: -2,-1,0,1,2
        pos_one = 1.0 * ((V > alpha/2.0) & (V < 3*alpha/2.0))
        pos_two = 2.0 * (V >= 3*alpha/2.0)
        neg_one = -1.0 * ((V < -alpha/2.0) & (V > -3*alpha/2.0))
        neg_two = -2.0 * (V <= -3*alpha/2.0)
        Q = pos_one + pos_two + neg_one + neg_two

    elif kbits == 7:
        # Levels: -4,-2,-1,0,1,2,4
        pos_one = 1.0*((V > alpha/2.) & (V < 3*alpha/2.))
        pos_two = 2.0*((V >= 3*alpha/2.) & (V < 3*alpha))
        pos_four= 4.0*(V >= 3*alpha)
        neg_one = -1.0*((V < -alpha/2.) & (V > -3*alpha/2.))
        neg_two = -2.0*((V <= -3*alpha/2.) & (V > -3*alpha))
        neg_four= -4.0*(V <= -3*alpha)
        Q = pos_one + pos_two + pos_four + neg_one + neg_two + neg_four

    elif kbits == 9:
        # Levels: -8, -4, -2, -1, 0, 1, 2, 4, 8
        # thresholds at alpha*(0.5,1.5,3,6)
        pos_one   =  1.0*((V > 0.5*alpha ) & (V < 1.5*alpha ))
        pos_two   =  2.0*((V >=1.5*alpha ) & (V < 3*alpha   ))
        pos_four  =  4.0*((V >=3*alpha   ) & (V < 6*alpha   ))
        pos_eight =  8.0*( V >=6*alpha   )
        neg_one   = -1.0*((V < -0.5*alpha) & (V > -1.5*alpha))
        neg_two   = -2.0*((V <=-1.5*alpha) & (V > -3*alpha  ))
        neg_four  = -4.0*((V <=-3*alpha  ) & (V > -6*alpha  ))
        neg_eight = -8.0*( V <=-6*alpha  )
        Q = (pos_one+pos_two+pos_four+pos_eight 
             +neg_one+neg_two+neg_four+neg_eight)

    elif kbits == 11:
        # -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16
        pos_one     =  1.0*((V > 0.5*alpha ) & (V <=1.5*alpha ))
        pos_two     =  2.0*((V > 1.5*alpha ) & (V <=3.0*alpha ))
        pos_four    =  4.0*((V > 3.0*alpha ) & (V <=6.0*alpha ))
        pos_eight   =  8.0*((V > 6.0*alpha ) & (V <=12.*alpha ))
        pos_sixteen = 16.0*( V >12.*alpha )
        neg_one     = -1.0*((V < -0.5*alpha ) & (V >= -1.5*alpha ))
        neg_two     = -2.0*((V < -1.5*alpha ) & (V >= -3.0*alpha ))
        neg_four    = -4.0*((V < -3.0*alpha ) & (V >= -6.0*alpha ))
        neg_eight   = -8.0*((V < -6.0*alpha ) & (V >= -12.*alpha))
        neg_sixteen = -16.0*( V < -12.*alpha)
        Q = (pos_one + pos_two + pos_four + pos_eight + pos_sixteen
             +neg_one +neg_two +neg_four +neg_eight +neg_sixteen)

    else:
        raise NotImplementedError("kbits must be one of 3,5,7,9,11.")
    return Q

def mapping(Q, alpha, beta=0, kbits=3):
    """
    Mapping Q in {..., -4, -2, -1, 0, 1, 2, 4, ...} to real values G = alpha*Q.
    """
    G = alpha * Q
    return G

##############################
# 3) Projection & ADMM
##############################

def symmetric_projection(V, weight, hessian, init_alpha, kbits=3, layer_name='unknown',
                         error_info_file=None, proj_ite_times=10):
    """
    Iteratively refine alpha by projecting V to a discrete set, i.e. Q = quantize(V, alpha),
    alpha = (V dot Q)/(Q dot Q). Return best G = alpha*Q with minimal MSE.
    """
    last_alpha = init_alpha
    alpha = init_alpha
    best_alpha = alpha

    # Convert to numpy if needed
    if isinstance(V, torch.Tensor):
        V_np = V.cpu().numpy()
    else:
        V_np = V
    Q = quantize(V_np, alpha, kbits=kbits)
    best_G = mapping(Q, alpha, 0, kbits)
    min_error = np.sum((V_np - best_G)**2)

    if error_info_file:
        error_info_file.write('[Before Projection] MSE: %f, alpha=%f\n'%(min_error,best_alpha))

    for proj_ite in range(proj_ite_times):
        Q = quantize(V_np, alpha, kbits=kbits)
        numerator   = np.sum(V_np.flatten()*Q.flatten())
        denominator = np.sum(Q.flatten()*Q.flatten())+1e-9
        alpha_new   = numerator/denominator

        G_new = mapping(Q, alpha_new, 0, kbits)
        mse_new= np.sum((V_np - G_new)**2)

        if error_info_file:
            error_info_file.write('[Proj ite %d] MSE=%f, alpha=%f\n'%(proj_ite,mse_new,alpha_new))

        if mse_new<min_error:
            min_error = mse_new
            best_alpha= alpha_new
            best_G    = G_new

        if abs(last_alpha-alpha_new)<1e-4:
            break
        else:
            last_alpha = alpha_new

        alpha = alpha_new

    return best_G

# def ADMM_quantization(layer_name, layer_type, kernel, bias, hessian, kbits, stat_dict=None,
#                       error_info_root=None, save_root=None, rho_factor=1, 
#                       ADMM_ite_times=100, proj_method='LDNQ'):
#     """
#     ADMM-based LDNQ quantization method. 
#     'layer_type' can be 'C' (conv with bias), 'F' (fully connected with bias), 'R'(res layer?), 'F-'.
#     'proj_method' can be 'LDNQ', 'kmeans', 'dorefa', etc.
#     """
#     # 0) Unfold or flatten
#     if layer_type == 'C':
#         # conv with bias
#         weight_unfold = unfold_kernel(kernel)
#         kernel_shape  = kernel.shape
#         W = np.concatenate([weight_unfold, bias.reshape(1, -1)], axis=0)  # [in_features+1, out_features]
#     elif layer_type == 'R':
#         weight_unfold = unfold_kernel(kernel)
#         kernel_shape  = kernel.shape
#         W = weight_unfold
#     elif layer_type == 'F':
#         # fully connected with bias
#         W = np.concatenate([kernel.transpose(), bias.reshape(1, -1)], axis=0)
#     else:
#         # 'F-' or res
#         W = kernel.transpose()

#     l1, l2 = W.shape

#     # 1) init G
#     if proj_method == 'kmeans':
#         enc    = OneHotEncoder()
#         km     = KMeans(n_clusters=kbits).fit(W.reshape([-1,1]))
#         label_ = enc.fit_transform(km.labels_.reshape([-1,1]))
#         G_init = label_.dot(km.cluster_centers_).reshape(W.shape)
#     elif proj_method == 'dorefa':
#         G_init = dorefa_fw(torch.from_numpy(W), bitW=kbits).numpy()
#     else:
#         alpha_init= np.mean(np.abs(W))
#         G_init   = alpha_init*quantize(W, alpha_init, kbits=kbits)

#     G = G_init
#     dual= np.zeros_like(W)
#     rho = rho_factor*hessian[0,0] if hessian is not None else rho_factor

#     A   = hessian + rho*np.eye(l1) if hessian is not None else rho*np.eye(l1)
#     b_1 = np.dot(hessian, W) if hessian is not None else np.zeros_like(W)

#     # track best
#     ADMM_min_error = get_error(G, hessian, W) if hessian is not None else 1e20
#     best_G = G

#     ascend_count=0

#     for admm_it in range(ADMM_ite_times):
#         # Proximal Step
#         b = b_1 + rho*(G - dual)
#         try:
#             W_new = np.linalg.solve(A,b)
#         except:
#             W_new = np.linalg.lstsq(A,b,rcond=None)[0]

#         # Projection Step
#         V = W_new+dual
#         if proj_method=='kmeans':
#             # re-run kmeans
#             km     = KMeans(n_clusters=kbits).fit(V.reshape([-1,1]))
#             enc    = OneHotEncoder()
#             label_ = enc.fit_transform(km.labels_.reshape([-1,1]))
#             G_proj = label_.dot(km.cluster_centers_).reshape(V.shape)
#         elif proj_method=='dorefa':
#             G_proj = dorefa_fw(torch.from_numpy(V), bitW=kbits).numpy()
#         else:
#             alpha_approx= np.mean(np.abs(V))
#             G_proj      = symmetric_projection(V, W, hessian, alpha_approx, kbits)

#         # measure error
#         cur_error = get_error(G_proj, hessian, W) if hessian is not None else np.sum((W - G_proj)**2)

#         # dual update
#         dual+= (W_new - G_proj)
#         # check best
#         if cur_error<ADMM_min_error:
#             ADMM_min_error=cur_error
#             best_G= G_proj
#             ascend_count=0
#         else:
#             ascend_count+=1

#         if ascend_count>=3:
#             break

#         G=G_proj
#         W_new=None

#     # 2) reshape best_G back
#     if layer_type=='C':
#         # [in_features+1, out_features]
#         if bias is not None:
#             kernel_fold= fold_weights(best_G[0:-1,:], kernel_shape)
#             bias_fold  = best_G[-1,:].flatten()
#             return kernel_fold, bias_fold
#         else:
#             kernel_fold= fold_weights(best_G, kernel_shape)
#             return kernel_fold
#     elif layer_type=='R':
#         kernel_fold= fold_weights(best_G, kernel_shape)
#         return kernel_fold
#     elif layer_type=='F':
#         best_G = best_G.transpose() # [out_features, in_features+1]
#         w_ = best_G[:,0:-1]
#         b_ = best_G[:,-1].flatten()
#         return w_, b_
#     else:
#         # F-
#         best_G = best_G.transpose()
#         return best_G
def ADMM_quantization(layer_name, layer_type, kernel, bias, hessian, kbits,
                      stat_dict=None, error_info_root=None, save_root=None,
                      rho_factor=1, ADMM_ite_times=100, proj_method='LDNQ'):
    """
    ADMM-based LDNQ quantization method from the paper's source code.
    layer_type in {'C','R','F','F-'}:
      - 'C': conv with bias (kernel shape=4D, bias!=None)
      - 'R': conv with no bias
      - 'F': fully-connected with bias
      - 'F-': fully-connected w/o bias
    kernel, bias: numpy arrays
    hessian: if you have second-order info, pass it; else None
    kbits: discrete levels
    ...
    """
    # Unfold or flatten
    if layer_type == 'C':
        # Conv *with* bias
        # If your actual layer truly has no bias, pass layer_type='R' or set bias!=None here.
        weight = unfold_kernel(kernel)
        kernel_shape = kernel.shape
        if bias is None:
            # Graceful fix: treat it as 'R'
            W = weight
        else:
            W = np.concatenate([weight, bias.reshape(1, -1)], axis=0)

    elif layer_type == 'R':
        # Conv without bias
        weight = unfold_kernel(kernel)
        kernel_shape = kernel.shape
        W = weight

    elif layer_type == 'F':
        # Fully-connected with bias
        W = np.concatenate([kernel.transpose(), bias.reshape(1, -1)], axis=0)
    elif layer_type == 'F-':
        # Fully-connected no bias
        W = kernel.transpose()
    else:
        raise NotImplementedError(f"Unknown layer_type: {layer_type}")

    l1, l2 = W.shape

    # 1) Initialize G
    if proj_method == 'kmeans':
        enc = OneHotEncoder()
        kmeans = KMeans(n_clusters=kbits).fit(W.reshape([-1, 1]))
        label_oht = enc.fit_transform(kmeans.labels_.reshape([-1,1]))
        G = label_oht.dot(kmeans.cluster_centers_).reshape(W.shape)
    elif proj_method == 'dorefa':
        G = dorefa_fw(torch.from_numpy(W), bitW=kbits).numpy()
    else:
        alpha_init = np.mean(np.abs(W))
        G = alpha_init * quantize(W, alpha_init, kbits=kbits)

    dual = np.zeros_like(W)
    if hessian is not None:
        rho = rho_factor * hessian[0, 0]
        A = hessian + rho*np.eye(l1)
        b_1 = np.dot(hessian, W)
    else:
        rho = rho_factor
        A = rho*np.eye(l1)
        b_1= np.zeros_like(W)

    # track best
    if hessian is not None:
        ADMM_min_error = get_error(G, hessian, W)
    else:
        ADMM_min_error = np.sum((W - G)**2)
    best_G = G

    ascend_count=0

    for ADMM_it in range(ADMM_ite_times):
        # Proximal
        b = b_1 + rho*(G - dual)
        try:
            W_star = np.linalg.solve(A, b)
        except:
            W_star = np.linalg.lstsq(A,b,rcond=None)[0]

        # Projection
        V = W_star + dual
        if proj_method == 'kmeans':
            km = KMeans(n_clusters=kbits).fit(V.reshape([-1,1]))
            enc = OneHotEncoder()
            label_ = enc.fit_transform(km.labels_.reshape([-1,1]))
            G_proj= label_.dot(km.cluster_centers_).reshape(V.shape)
        elif proj_method == 'dorefa':
            G_proj = dorefa_fw(torch.from_numpy(V), bitW=kbits).numpy()
        else:
            alpha_est = np.mean(np.abs(V))
            G_proj= symmetric_projection(V, W, hessian, alpha_est, kbits)

        # measure error
        if hessian is not None:
            cur_error = get_error(G_proj, hessian, W)
        else:
            cur_error= np.sum((W - G_proj)**2)

        # dual update
        dual += (W_star - G_proj)

        # check best
        if cur_error<ADMM_min_error - 1e-6:
            ADMM_min_error= cur_error
            best_G= G_proj.copy()
            ascend_count=0
        else:
            ascend_count+=1
        if ascend_count>=3:
            break
        G = G_proj

    # 2) Reshape best_G
    if layer_type=='C':
        if bias is None:
            # no bias => 'R' logic
            quantized_kernel = fold_weights(best_G, kernel_shape)
            return quantized_kernel, None
        else:
            quantized_kernel = fold_weights(best_G[0:-1,:], kernel_shape)
            quantized_bias   = best_G[-1,:].flatten()
            return quantized_kernel, quantized_bias

    elif layer_type=='R':
        quantized_kernel = fold_weights(best_G, kernel_shape)
        return quantized_kernel, None

    elif layer_type=='F':
        best_G= best_G.transpose() # [outF, inF+1]
        w_q= best_G[:,0:-1]
        b_q= best_G[:,-1].flatten()
        return w_q, b_q

    elif layer_type=='F-':
        best_G= best_G.transpose()
        return best_G, None


##############################
# 4) Additional Methods: Direct, DoReFa, etc.
##############################

def direct_quantize(param, kbits=3):
    """
    Simple approach: alpha = mean(|param|)
    Q = quantize(param, alpha, kbits)
    G = alpha*Q
    """
    if isinstance(param, torch.Tensor):
        param_np = param.detach().cpu().numpy()
    else:
        param_np = param
    alpha_val = np.mean(np.abs(param_np))
    Q_np = quantize(param_np, alpha_val, kbits=kbits)
    G_np = mapping(Q_np, alpha_val, 0, kbits)
    if isinstance(param, torch.Tensor):
        G = torch.from_numpy(G_np).to(param.device, param.dtype)
        return G
    else:
        return G_np

def dorefa_quantize(param, kbits=8):
    """
    Basic DoReFa quantization: scale param in [0,1], round to discrete steps, scale back.
    """
    n = float(2**kbits - 1)
    return torch.round(param*n)/n

def dorefa_fw(param, bitW=8):
    """
    DoReFa weight quantization function: 
    w -> tanh(w)/max|w| -> [0,1] -> discrete -> [-1,1]
    """
    if isinstance(param, np.ndarray):
        param_t = torch.from_numpy(param).float()
    else:
        param_t = param.float()
    x = torch.tanh(param_t)
    x = x/ torch.max(torch.abs(x)) *0.5 +0.5
    x_q = dorefa_quantize(x, bitW)
    w_q = 2.*x_q-1.
    if isinstance(param, np.ndarray):
        return w_q.cpu().numpy()
    else:
        return w_q.type_as(param)


print("\n[Cell 3 Complete] - LDNQ/ADMM quantization + additional methods (kmeans, dorefa, direct) loaded.")


# ============================
# Cell 4: Full LDNQ Cascade + Retraining, for All Models
# ============================

import torch
import torch.nn as nn
import torch.optim as optim
import time, os, random
import numpy as np

###########################################
# A) 1% Data Subset Creation (CIFAR & Tiny)
###########################################
def create_small_subset(trainloader_full, fraction=0.01):
    """
    Given a full training dataloader, extract ~ fraction of the dataset
    to build a smaller DataLoader for mandatory retraining.
    """
    from torch.utils.data import Subset
    dataset = trainloader_full.dataset
    total_len = len(dataset)
    small_len = int(total_len * fraction)
    indices = list(range(total_len))
    random.shuffle(indices)
    subset_inds = indices[:small_len]

    small_dataset = Subset(dataset, subset_inds)
    small_loader = torch.utils.data.DataLoader(
        small_dataset, 
        batch_size=trainloader_full.batch_size,
        shuffle=True,
        num_workers=2
    )
    return small_loader

# Create 1% subset for CIFAR-10 if trainloader_cifar is present
trainloader_small_cifar = None
if 'trainloader_cifar' in globals():
    trainloader_small_cifar = create_small_subset(trainloader_cifar, fraction=0.01)
    print(f"[CIFAR-10] 1% subset created => {len(trainloader_small_cifar.dataset)} samples.")
else:
    print("trainloader_cifar not found or not defined. Skipping CIFAR subset.")

# Create 1% subset for TinyImageNet if trainloader_imagenet is present
trainloader_small_tiny = None
if 'trainloader_imagenet' in globals() and trainloader_imagenet is not None:
    trainloader_small_tiny = create_small_subset(trainloader_imagenet, fraction=0.01)
    print(f"[TinyImageNet] 1% subset created => {len(trainloader_small_tiny.dataset)} samples.")
else:
    print("trainloader_imagenet not found or None. Skipping Tiny subset.")


###########################################
# B) Measurement Utilities
###########################################
def measure_accuracy(model, dataloader, device):
    if dataloader is None:
        return None
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    return 100.0 * correct / total

def measure_inference_time(model, dataloader, device, n_runs=5):
    if dataloader is None:
        return None
    model.eval()
    times=[]
    count=0
    with torch.no_grad():
        for images,_ in dataloader:
            images = images.to(device)
            start = time.time()
            _ = model(images)
            end   = time.time()
            times.append(end-start)
            count+=1
            if count>=n_runs:
                break
    return np.mean(times) if len(times)>0 else None

def measure_model_size(model):
    total_params=0
    for p in model.parameters():
        total_params += p.numel()
    # float32 => 4 bytes/param
    return (total_params*4)/(1024**2)


###########################################
# C) Layerwise LDNQ with Mandatory Retraining
###########################################
def layerwise_LDNQ_cascade(model, trainloader_small, device, kbits=3, n_epochs=1, lr=1e-4):
    """
    Paper's LDNQ (ADMM) approach:
    For each quantizable layer i:
      1) Quantize it (ADMM).
      2) Freeze that layer's params.
      3) Retrain the rest (in full precision) on the 1% subset for 'n_epochs'.
    Repeats until all conv/FC layers are quantized.
    """
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam

    # from prior cell (Cell 3) we have ADMM_quantization
    global ADMM_quantization
    if ADMM_quantization is None:
        raise RuntimeError("ADMM_quantization not found. Make sure Cell 3 with ADMM code is loaded.")

    # freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # collect conv/fc param references
    param_list = []
    for name, param in model.named_parameters():
        if ("bn" not in name) and ("bias" not in name):
            param_list.append((name, param))

    ce_loss = CrossEntropyLoss()

    # ADMM quant for each param
    for idx, (layer_name, param) in enumerate(param_list):
        print(f"\n[LDNQ] Layer {idx+1}/{len(param_list)} quantizing => {layer_name}")
        # convert to numpy
        param_data = param.detach().cpu().numpy()
        shape_ = param_data.shape

        # Decide layer type: 'C' if shape=4D (conv), 'F-' if shape=2D (fc).
        if len(shape_)==4:
            ltype='C'
            quant_kern, _bias = ADMM_quantization(
                layer_name, ltype, param_data, None,  # no bias
                hessian=None, # we skip Hessian
                kbits=kbits, 
                proj_method='LDNQ',
                ADMM_ite_times=20
            )
            with torch.no_grad():
                param.copy_(torch.from_numpy(quant_kern).to(param.device, param.dtype))
        elif len(shape_)==2:
            ltype='F-'
            quant_w,_ = ADMM_quantization(
                layer_name, ltype, param_data, None, 
                hessian=None,
                kbits=kbits,
                proj_method='LDNQ',
                ADMM_ite_times=20
            )
            with torch.no_grad():
                param.copy_(torch.from_numpy(quant_w).to(param.device, param.dtype))
        else:
            print(f"Skipping unknown layer shape {shape_}")
            continue

        # freeze this newly quantized param, unfreeze the rest for retraining
        for (n,p) in model.named_parameters():
            if n==layer_name:
                p.requires_grad=False
            else:
                if ("bn" not in n) and ("bias" not in n):
                    p.requires_grad=True
                else:
                    p.requires_grad=False

        # retrain with 1% data for n_epochs
        if trainloader_small is not None and n_epochs>0:
            model.train()
            opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            for ep in range(n_epochs):
                total_loss=0.0
                count=0
                for images, labels in trainloader_small:
                    images, labels = images.to(device), labels.to(device)
                    opt.zero_grad()
                    out = model(images)
                    loss = ce_loss(out, labels)
                    loss.backward()
                    opt.step()
                    total_loss+=loss.item()
                    count+=1
                if count>0:
                    avg_loss= total_loss/count
                else:
                    avg_loss=0
                print(f"   Retrain epoch {ep+1}/{n_epochs}, Loss={avg_loss:.4f}")

        # freeze everything again after retrain
        for p in model.parameters():
            p.requires_grad=False

    print("[LDNQ] All layers quantized + mandatory retraining done.")
    return model

###########################################
# D) Direct or DoReFa Quant + Mandatory Retraining (layerwise)
###########################################
def layerwise_other_methods_cascade(model, trainloader_small, device, method="direct", kbits=3, n_epochs=1, lr=1e-4):
    """
    Similar to LDNQ approach, but uses either direct quant or doReFa for each layer,
    then retrains the rest. 
    """
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam

    # from prior cell code
    global direct_quantize, dorefa_fw
    if direct_quantize is None or dorefa_fw is None:
        raise RuntimeError("direct_quantize / dorefa_fw not found. Ensure Cell 3 is loaded.")

    for p in model.parameters():
        p.requires_grad=False

    param_list=[]
    for name,param in model.named_parameters():
        if ("bn" not in name) and ("bias" not in name):
            param_list.append((name, param))

    ce_loss = CrossEntropyLoss()

    for idx, (layer_name, param) in enumerate(param_list):
        print(f"\n[{method}] Layer {idx+1}/{len(param_list)} => {layer_name}")
        # quant
        with torch.no_grad():
            param_data = param.detach().cpu().numpy()
        if method=="direct":
            new_val = direct_quantize(param, kbits=kbits)
            with torch.no_grad():
                param.copy_(new_val)
        elif method=="dorefa":
            q_np = dorefa_fw(param_data, bitW=kbits)
            with torch.no_grad():
                param.copy_(torch.from_numpy(q_np).to(param.device, param.dtype))
        else:
            print("Unknown method:",method)
            continue

        # freeze this param, unfreeze others
        for (n,p_) in model.named_parameters():
            if n==layer_name:
                p_.requires_grad=False
            else:
                if ("bn" not in n) and ("bias" not in n):
                    p_.requires_grad=True
                else:
                    p_.requires_grad=False

        # retrain
        if trainloader_small is not None and n_epochs>0:
            model.train()
            opt = Adam(filter(lambda p_: p_.requires_grad, model.parameters()), lr=lr)
            for ep in range(n_epochs):
                total_loss=0.0
                c=0
                for images, labels in trainloader_small:
                    images, labels = images.to(device), labels.to(device)
                    opt.zero_grad()
                    out= model(images)
                    loss= ce_loss(out, labels)
                    loss.backward()
                    opt.step()
                    total_loss+=loss.item()
                    c+=1
                if c>0:
                    avg_loss= total_loss/c
                else:
                    avg_loss=0
                print(f"   Retrain epoch {ep+1}/{n_epochs}, Loss={avg_loss:.4f}")

        # freeze everything again
        for p_ in model.parameters():
            p_.requires_grad=False

    print(f"[{method}] All layers quantized + mandatory retraining done.")
    return model

###########################################
# E) Main Experiments for All Models
###########################################
def run_all_quant_experiments(
    pretrained_models_dir,
    device,
    testloader_cifar=None,
    testloader_imagenet=None,
    trainloader_small_cifar=None,
    trainloader_small_tiny=None,
    bits_list=[3,5,7],
    methods=["LDNQ","direct","dorefa"]
):
    """
    This function:
      1) Lists all your models:
         - ImageNet: resnet18, resnet34, resnet50, alexnet
         - CIFAR-10: resnet20, resnet32, resnet56, vgg16, wrn
      2) For each model => measure baseline => for each method => for each bits => layerwise quant+retrain => measure final => save
    """
    # Build a list describing each model
    from functools import partial

    # we assume you have definitions: 
    #  resnet20_cifar, resnet32_cifar, resnet56_cifar, vgg16_cifar, wrn50_cifar
    #  load_imagenet_resnet18, etc., or you can define partial.
    global resnet20_cifar, resnet32_cifar, resnet56_cifar, vgg16_cifar, wrn50_cifar
    global load_imagenet_resnet18, load_imagenet_resnet34, load_imagenet_resnet50, load_imagenet_alexnet

    model_info = [
        # (model_name, isCIFAR, constructor_fn, checkpoint_path)
        # ("resnet18",    False, load_imagenet_resnet18, os.path.join(pretrained_models_dir,"resnet18.pth")),
        # ("resnet34",    False, load_imagenet_resnet34, os.path.join(pretrained_models_dir,"resnet34.pth")),
        # ("resnet50",    False, load_imagenet_resnet50, os.path.join(pretrained_models_dir,"resnet50.pth")),
        # ("alexnet",     False, load_imagenet_alexnet,  os.path.join(pretrained_models_dir,"alexnet.pth")),

        ("resnet20",    True,  resnet20_cifar,         os.path.join(pretrained_models_dir,"resnet20.pth")),
        ("resnet32",    True,  resnet32_cifar,         os.path.join(pretrained_models_dir,"resnet32.pth")),
        ("resnet56",    True,  resnet56_cifar,         os.path.join(pretrained_models_dir,"resnet56.pth")),
        ("vgg16",       True,  vgg16_cifar,            os.path.join(pretrained_models_dir,"vgg16.pth")),
        ("wrn",         True,  wrn50_cifar,            os.path.join(pretrained_models_dir,"wrn.pth")),
    ]

    # We'll store results in a list of [model, method, bits, acc, time, size].
    results = []
    os.makedirs("quantized_models", exist_ok=True)

    for (mname, isCIFAR, constructor_fn, ckpt_path) in model_info:
        print(f"\n==== Now Processing Model: {mname} ====")
        # 1) Load
        net = constructor_fn()
        state_dict = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()

        # choose test loader
        testloader_ = testloader_cifar if isCIFAR else testloader_imagenet
        # choose small train loader
        trainloader_small_ = trainloader_small_cifar if isCIFAR else trainloader_small_tiny

        # 2) measure baseline
        base_acc  = measure_accuracy(net, testloader_, device)
        base_time = measure_inference_time(net, testloader_, device, n_runs=3)
        base_size = measure_model_size(net)
        print(f"[Baseline] {mname} => Acc={base_acc}, InfTime={base_time}, Size={base_size:.2f}MB")
        results.append([mname,"baseline","-", base_acc, base_time, base_size])

        # 3) For each method & bit
        for method in methods:
            for kb in bits_list:
                print(f"\n>>> {mname}: {method}, {kb}-bits quantization + mandatory retraining.")
                # reload fresh
                net_q = constructor_fn()
                net_q.load_state_dict(torch.load(ckpt_path, map_location=device))
                net_q.to(device)
                net_q.eval()

                if method=="LDNQ":
                    net_q = layerwise_LDNQ_cascade(
                        net_q, 
                        trainloader_small=trainloader_small_,
                        device=device,
                        kbits=kb,
                        n_epochs=1,   # you can adjust
                        lr=1e-4
                    )
                elif method in ["direct","dorefa"]:
                    net_q = layerwise_other_methods_cascade(
                        net_q, 
                        trainloader_small=trainloader_small_,
                        device=device,
                        method=method,
                        kbits=kb,
                        n_epochs=1,
                        lr=1e-4
                    )
                else:
                    print(f"Unknown method: {method}, skipping.")
                    continue

                # measure final
                final_acc  = measure_accuracy(net_q, testloader_, device)
                final_time = measure_inference_time(net_q, testloader_, device, n_runs=3)
                final_size = measure_model_size(net_q)
                print(f"=> {mname}[{method},{kb}-bits] => Acc={final_acc}, Time={final_time}, Size={final_size:.2f}MB")
                results.append([mname, method, f"{kb}-bits", final_acc, final_time, final_size])

                # save model
                q_save_path = os.path.join("quantized_models", f"{mname}_{method}_{kb}bits.pth")
                torch.save(net_q.state_dict(), q_save_path)
                print(f"Quantized model saved => {q_save_path}")
    
    print("\n=== Final Results ===")
    for row in results:
        print(row)
    return results

print("""
[Cell 4 Complete] 
This single cell:
1) Creates a 1% subset for mandatory retraining (CIFAR & TinyImageNet).
2) Defines measure_* functions for accuracy, time, size.
3) Implements 'layerwise_LDNQ_cascade' for ADMM-based quantization with forced retraining after each layer.
4) Implements 'layerwise_other_methods_cascade' for direct/doReFa with forced retraining too.
5) 'run_all_quant_experiments' loops over the 9 models, for each method & bits, performs layerwise quant + retraining, logs & saves results.
""")


results = run_all_quant_experiments(
    pretrained_models_dir=pretrained_models_dir,
    device=device,
    testloader_cifar=testloader_cifar,
    testloader_imagenet=testloader_imagenet,
    trainloader_small_cifar=trainloader_small_cifar,
    trainloader_small_tiny=trainloader_small_tiny,
    bits_list=[3,5,7], 
    methods=["LDNQ","direct","dorefa"]
)
