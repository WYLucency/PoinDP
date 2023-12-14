"""Math utils functions."""

import torch

# import torch
from torch.nn import functional as F
from torch.distributions import Normal, Independent
from numbers import Number
from torch.distributions.utils import _standard_normal, broadcast_all
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class WrappedNormal(torch.distributions.Distribution):

    arg_constraints = {'loc': torch.distributions.constraints.real,
                       'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        raise NotImplementedError

    @property
    def scale(self):
        return F.softplus(self._scale) if self.softplus else self._scale

    def __init__(self, loc, scale, manifold, validate_args=None, softplus=False):
        self.dtype = loc.dtype
        self.softplus = softplus
        self.loc, self._scale = broadcast_all(loc, scale)
        self.manifold = manifold
        self.manifold.assert_check_point_on_manifold(self.loc)
        self.device = loc.device
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape, event_shape = torch.Size(), torch.Size()
        else:
            batch_shape = self.loc.shape[:-1]
            event_shape = torch.Size([self.manifold.dim])
        super(WrappedNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, shape=torch.Size(), sigma=torch.Tensor):
        with torch.no_grad():
            return self.rsample(shape,sigma)

    def rsample(self, sample_shape=torch.Size(), sigma=torch.Tensor):
        shape = self._extended_shape(sample_shape)
        v = sigma * _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        # v = sigma.unsqueeze(2) * _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        self.manifold.assert_check_vector_on_tangent(self.manifold.zero, v)
        v = v / self.manifold.lambda_x(self.manifold.zero, keepdim=True)
        u = self.manifold.transp(self.manifold.zero, self.loc, v)
        z = self.manifold.expmap(self.loc, u)
        return z


def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    return x.double().clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        z = x.double().clamp(-1 + 1e-2, 1 - 1e-2)
        ctx.save_for_backward(z)
        return (torch.log_((1 + z).clamp_min(1e-8)).sub_(torch.log_((1 - z).clamp_min(1e-8)))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5
    
def split_idx(samples, train_size, val_size, random_state=None):
    train, val = train_test_split(samples, train_size=train_size, random_state=random_state)
    if isinstance(val_size, float):
        val_size *= len(samples) / len(val)
    val, test = train_test_split(val, train_size=val_size, random_state=random_state)
    return train, val, test
    
def split_idx1(samples1, samples2, train_size, val_size, random_state=None):
    train, val = train_test_split(samples1, train_size=train_size, random_state=random_state)
    val = torch.cat((val,samples2))
    val, test = train_test_split(val, train_size=val_size, random_state=random_state)
    return train, val, test

def micro_macro_f1_score(logits, labels):
    prediction = torch.argmax(logits, dim=1).cpu().long().numpy()
    labels = labels.cpu().numpy() 
    micro_f1 = f1_score(labels, prediction, average='micro')
    weighted_f1 = f1_score(labels, prediction, average='weighted')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return micro_f1, weighted_f1, macro_f1