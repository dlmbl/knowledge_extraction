import torch
from functools import partial

def hook_fn(in_grads, out_grads, m, i, o):
  for grad in i:
    try:
      in_grads.append(grad)
    except AttributeError: 
      pass

  for grad in o:  
    try:
      out_grads.append(grad.cpu().numpy())
    except AttributeError: 
      pass
    
def get_gradients_from_layer(net, x, y, layer_name=None, normalize=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xx = torch.tensor(x, device=device).unsqueeze(0)
    yy = torch.tensor([y], device=device)
    xx = xx.unsqueeze(0)
    in_grads = []
    out_grads = []
    try:
        for param in net.features.parameters():
            param.requires_grad = True
    except AttributeError:
        for param in net.parameters():
            param.requires_grad = True

    if layer_name is None:
        layers = [(name,module) for name, module in net.named_modules() if type(module) == torch.nn.Conv2d][-1]
        layer_name = layers[0]
        layer = layers[1]
    else:
        layers = [module for name, module in net.named_modules() if name == layer_name]
        assert(len(layers) == 1)
        layer = layers[0]

    layer.register_backward_hook(partial(hook_fn, in_grads, out_grads))

    out = net(xx)
    out[0][y].backward()
    grad = out_grads[0]
    if normalize:
        max_grad = np.max(np.abs(grad))
        if max_grad>10**(-12):
            grad /= max_grad 
        else:
            grad = np.zeros(np.shape(grad))

    return grad
