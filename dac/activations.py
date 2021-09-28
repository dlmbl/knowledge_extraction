import numpy as np
import torch
import torch.nn as nn
from functools import partial
import cv2

from dac.utils import image_to_tensor

def get_layer_names(net):
    layer_names = util.get_model_layers(net, False)
    # print(layer_names)

def save_activation(activations, name, mod, inp, out):
    activations[name].append(out.cpu())
    
def get_activation_dict(net, images, activations):
    """
    net: The NN object
    images: list of 2D (h,w) normalized image arrays.
    """
    tensor_images = []
    for im in images:
        tensor_images.append(image_to_tensor(im))
    
    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    for name, m in net.named_modules():
        if type(m)==nn.Conv2d or type(m) == nn.Linear:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, activations, name))

    # forward pass through the full dataset
    out = []
    for tensor_image in tensor_images:
        out.append(net(tensor_image).detach().cpu().numpy())
        
    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations_dict = {name: torch.cat(outputs, 0).cpu().detach().numpy() for name, outputs in activations.items()}
    return activations_dict, out

def get_layer_activations(activations_dict, layer_name):
    layer_activation = None
    for name, activation in activations_dict.items():
        if name == layer_name:
            layer_activation = activation
    return layer_activation

def project_layer_activations_to_input_rescale(layer_activation, input_shape):
    """
    Projects the nth activation and the cth channel from layer 
    to input. layer_activation[n,c,:,:] -> Input
    """
    act_shape = np.shape(layer_activation)
    n = act_shape[0]
    c = act_shape[1]
    h = act_shape[2]
    w = act_shape[3]
    
    samples = [i for i in range(n)]
    channels = [c for c in range(c)]
    
    canvas = np.zeros([len(samples), len(channels), input_shape[0], input_shape[1]], 
                      dtype=np.float32)
    
    for n in samples:
        for c in channels:
            to_project = layer_activation[n,c,:,:]
            canvas[n,c,:,:] = cv2.resize(to_project, (input_shape[1], input_shape[0]))

    return canvas
