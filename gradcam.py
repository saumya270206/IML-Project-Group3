"""
Grad-CAM (Selvaraju et al., 2017) implemented from scratch in PyTorch.

    alpha_k^c = (1/Z) sum_{i,j} d y^c / d A^k_{ij}        (global-average-pooled gradients)
    L^c       = ReLU( sum_k alpha_k^c A^k )               (weighted combination + ReLU)

A = activations of the chosen conv layer, y^c = *pre-softmax* score of class c.
The coarse map L^c is bilinearly upsampled to the input resolution and min-max normalised.
"""
import torch
import torch.nn.functional as F

from utils import minmax_normalise


class GradCAM:
    def __init__(self, model, target_layer):
        """model: any nn.Module classifier; target_layer: the nn.Module whose output is A."""
        self.model = model
        self.target_layer = target_layer
        self._recording = False
        self._acts = None                      # output of target layer (kept WITH graph)
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        if self._recording:
            self._acts = output

    def remove(self):
        """Detach the forward hook (call when you are done with this object)."""
        self._handle.remove()

    def __call__(self, x, target=None):
        """
        x      : (B,3,H,W) normalised images
        target : (B,) class indices; None -> predicted class
        returns: (B,H,W) float map in [0,1] (detached)
        """
        was_training = self.model.training
        self.model.eval()                                   # BN in eval -> samples are independent
        x = x.detach().clone().requires_grad_(True)         # guarantees a graph even if params are frozen
        self._recording = True
        try:
            with torch.enable_grad():
                logits = self.model(x)
                if target is None:
                    target = logits.argmax(1)
                score = logits.gather(1, target[:, None]).sum()
                acts = self._acts                            # A, captured by the forward hook
                grads, = torch.autograd.grad(score, acts)    # d y^c / dA
        finally:
            self._recording = False
            self._acts = None
            self.model.train(was_training)
        acts, grads = acts.detach(), grads.detach()
        alpha = grads.mean(dim=(2, 3), keepdim=True)                   # (B,K,1,1) channel weights
        cam = F.relu((alpha * acts).sum(dim=1, keepdim=True))           # (B,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return minmax_normalise(cam)