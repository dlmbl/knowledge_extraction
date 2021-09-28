import collections
import numpy as np
import os
import torch

from dac.gradients import get_gradients_from_layer
from dac.activations import get_activation_dict, get_layer_activations, project_layer_activations_to_input_rescale
from dac.utils import normalize_image, save_image
from dac_networks import run_inference, init_network

def get_sgc(real_img, fake_img, real_class, fake_class, 
            net_module, checkpoint_path, input_shape, 
            input_nc, layer_name=None, output_classes=6, 
            downsample_factors=None):
    """
        real_img: Unnormalized (0-255) 2D image

        fake_img: Unnormalized (0-255) 2D image

        *_class: Index of real and fake class corresponding to network output

        net_module: Name of file and class name of the network to use. Must be placed in networks subdirectory

        checkpoint_path: Checkpoint of network.

        input_shape: Spatial input shape of network

        input_nc: Number of input channels.

        layer_name: Name of the conv layer to use (defaults to last)

        output_classes: Number of network output classes

        downsample_factors: Network downsample factors
    """


    if len(np.shape(fake_img)) != len(np.shape(real_img)) !=2:
        raise ValueError("Input images need to be two dimensional")
    
    imgs = [normalize_image(real_img), normalize_image(fake_img)]
    classes = [real_class, fake_class]

    if layer_name is None:
        net = init_network(checkpoint_path, input_shape, net_module, 
                           input_nc, eval_net=True, require_grad=False, 
                           output_classes=output_classes,
                           downsample_factors=downsample_factors)
        last_conv_layer = [(name,module) for name, module in net.named_modules() if type(module) == torch.nn.Conv2d][-1]
        layer_name = last_conv_layer[0]
        layer = last_conv_layer[1]
   
    grads = []
    for x,y in zip(imgs,classes):
        grad_net = init_network(checkpoint_path, input_shape, net_module, 
                                input_nc, eval_net=True, require_grad=False, 
                                output_classes=output_classes, 
                                downsample_factors=downsample_factors)
        grads.append(get_gradients_from_layer(grad_net, x, y, layer_name))

    acts_real = collections.defaultdict(list)
    acts_fake = collections.defaultdict(list)

    activation_net = init_network(checkpoint_path, input_shape, net_module, 
                                  input_nc, eval_net=True, require_grad=False, output_classes=output_classes,
                                  downsample_factors=downsample_factors)

    acts_real, out_real = get_activation_dict(activation_net, [imgs[0]], acts_real)
    acts_fake, out_fake = get_activation_dict(activation_net, [imgs[1]], acts_fake)

    acts = [acts_real, acts_fake]
    outs = [out_real, out_fake]
    
    layer_acts = []
    for act in acts:
        layer_acts.append(get_layer_activations(act, layer_name))

    delta_fake = grads[1] * (layer_acts[0] - layer_acts[1])
    delta_real = grads[0] * (layer_acts[1] - layer_acts[0])

    delta_fake_projected = project_layer_activations_to_input_rescale(delta_fake, (input_shape[0], input_shape[1]))[0,:,:,:]
    delta_real_projected = project_layer_activations_to_input_rescale(delta_real, (input_shape[0], input_shape[1]))[0,:,:,:]

    channels = np.shape(delta_fake_projected)[0]
    gc_0 = np.zeros(np.shape(delta_fake_projected)[1:])
    gc_1 = np.zeros(np.shape(delta_real_projected)[1:])

    for c in range(channels):
        gc_0 += delta_fake_projected[c,:,:]
        gc_1 += delta_real_projected[c,:,:]

    gc_0 = np.abs(gc_0)
    gc_1 = np.abs(gc_1)
    gc_0 /= np.max(np.abs(gc_0))
    gc_1 /= np.max(np.abs(gc_1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.tensor(gc_0, device=device), torch.tensor(gc_1, device=device)
