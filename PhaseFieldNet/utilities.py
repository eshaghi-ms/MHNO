import time

import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
from torch.utils.data import Dataset
import os

import operator
from functools import reduce
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


#################################################
#
# Utilities
#
#################################################
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        print(f"Available keys in self.data: {list(self.data.keys())}")
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


# normalization, point-wise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of nTrain*n or nTrain*T*n or nTrain*n*T in 1D
        # x could be in shape of nTrain*w*l or nTrain*T*w*l or nTrain*w*l*T in 2D
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last  # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            elif self.mean.ndim > sample_idx.ndim and not self.time_last:
                std = self.std[..., sample_idx] + self.eps  # T*batch*n
                mean = self.mean[..., sample_idx]
            else:
                std = None
                mean = None
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        my_min = torch.min(x, 0)[0].view(-1)
        my_max = torch.max(x, 0)[0].view(-1)

        self.a = (high - low) / (my_max - my_min)
        self.b = -self.a * my_max + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a is None:
            a = [1, ] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx // 2, step=1), torch.arange(start=-nx // 2, end=0, step=1)),
                        0).reshape(nx, 1).repeat(1, ny)
        k_y = torch.cat((torch.arange(start=0, end=ny // 2, step=1), torch.arange(start=-ny // 2, end=0, step=1)),
                        0).reshape(1, ny).repeat(nx, 1)
        k_x = torch.abs(k_x).reshape(1, nx, ny, 1).to(x.device)
        k_y = torch.abs(k_y).reshape(1, nx, ny, 1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced is False:
            weight = 1
            if k >= 1:
                weight += a[0] ** 2 * (k_x ** 2 + k_y ** 2)
            if k >= 2:
                weight += a[1] ** 2 * (k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
            weight = torch.sqrt(weight)
            loss = self.rel(x * weight, y * weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x ** 2 + k_y ** 2)
                loss += self.rel(x * weight, y * weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
                loss += self.rel(x * weight, y * weight)
            loss = loss / (k + 1)

        return loss


# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
    return c


class ImportDataset(Dataset):
    def __init__(self, parent_dir, matlab_dataset, normalized, T_in, T_out):
        self.y = None
        self.x = None
        self.T_in = T_in
        self.T_out = T_out
        self.normalized = normalized
        self.normalizer_x = None
        self.normalizer_y = None
        self.navier = True if "ns" in matlab_dataset else False

        matlab_dataset = parent_dir + matlab_dataset
        python_dataset = matlab_dataset.replace('.mat', '.pt')
        os.makedirs(parent_dir, exist_ok=True)

        if os.path.exists(python_dataset):
            print("Found saved dataset at", python_dataset)
            self.data = torch.load(python_dataset)['data']
        else:
            reader = MatReader(matlab_dataset)
            if self.navier:
                self.data = reader.read_field('u')
            else:
                self.data = reader.read_field('phi')
            torch.save({'data': self.data}, python_dataset)
        self.set_data()

    def set_data(self):
        if self.navier:
            self.x = self.data[:, *[slice(None)] * (self.data.ndim - 2), :self.T_in]
            self.y = self.data[:, *[slice(None)] * (self.data.ndim - 2), self.T_in:self.T_in + self.T_out]
        else:
            permute_order = list(range(self.data.ndim))
            permute_order.append(permute_order.pop(1))  # Move the second dimension to the end
            self.x = self.data[:, :self.T_in, *[slice(None)] * (self.data.ndim - 3)].permute(*permute_order)
            self.y = self.data[:, self.T_in:self.T_in + self.T_out, *[slice(None)] * (
                    self.data.ndim - 3)].permute(*permute_order)
        print(self.x.shape)
        print(self.y.shape)
        if self.normalized:
            self.make_normal()

    def make_normal(self):
        self.normalizer_x = UnitGaussianNormalizer(self.x)
        self.normalizer_y = UnitGaussianNormalizer(self.y)
        self.x = self.normalizer_x.encode(self.x)
        self.y = self.normalizer_y.encode(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ModelEvaluator:
    def __init__(self, model, test_dataset, s, T_in, T_out, device, normalized=False, normalizers=None,
                 time_history=False):
        self.model = model
        self.test_dataset = test_dataset
        self.s = s
        self.T_in = T_in
        self.T_out = T_out
        self.device = device
        self.normalized = normalized
        self.time_history = time_history
        self.normalizer_x = normalizers[0].to(self.device)
        self.normalizer_y = normalizers[1].to(self.device)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        spatial_dims = [s] * (len(test_dataset[0][0].shape) - 1)
        self.inp = torch.zeros((len(test_dataset), *spatial_dims, T_in))
        self.exact = torch.zeros((len(test_dataset), *spatial_dims, T_out))
        self.pred = torch.zeros((len(test_dataset), *spatial_dims, T_out))
        self.test_l2_set = []

    def evaluate(self, loss_fn):
        if self.time_history:
            index = 0
            step = 1
            with torch.no_grad():
                for xx, yy in self.test_loader:
                    self.inp[index] = xx.squeeze(0)
                    xx, yy = xx.to(self.device), yy.to(self.device)

                    for t in range(0, self.T_out, step):
                        # y = yy[..., t:t + step]
                        im = self.model(xx)
                        # loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)
                        xx = torch.cat((xx[..., step:], im), dim=-1)

                    self.exact[index] = yy.squeeze(0)
                    self.pred[index] = pred.squeeze(0)
                    test_l2 = loss_fn(pred.view(1, -1), yy.view(1, -1)).item()
                    self.test_l2_set.append(test_l2)
                    # print(index, test_l2)
                    index += 1

        else:
            index = 0
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    # out, Q, H = self.model(x)
                    t_start = time.time()
                    out = self.model(x)
                    if self.normalized:
                        out = self.normalizer_y.decode(out)
                        y = self.normalizer_y.decode(y)
                        x = self.normalizer_x.decode(x)
                    dt = time.time() - t_start
                    self.inp[index] = x.squeeze(0)
                    self.exact[index] = y.squeeze(0)
                    self.pred[index] = out.squeeze(0)
                    test_l2 = loss_fn(out.view(1, -1), y.view(1, -1)).item()
                    self.test_l2_set.append(test_l2)
                    print(index, dt, test_l2)
                    index += 1

                # phi = self.normalizer_y.decode(Q+H)
                # diff = self.normalizer_y.decode(Q-H)
                # Qn = self.normalizer_y.decode(Q)
                # Hn = self.normalizer_y.decode(H)
                # Q_exp_portion = torch.exp(Q.abs())/(torch.exp(Q.abs())+torch.exp(H.abs()))
                # H_exp_portion = torch.exp(H.abs())/(torch.exp(Q.abs())+torch.exp(H.abs()))
                # diff_exp = Q_exp_portion - H_exp_portion
                # AbsHportion = torch.abs(H) / (torch.abs(H) + torch.abs(Q))

                # field_names = ['Q', 'Hn', 'phi', 'diff', 'QExpPortion', 'HExpPortion', 'diff_exp', 'AbsHportion']
                # field_values = [Qn, Hn, phi, diff, Q_exp_portion, H_exp_portion, diff_exp, AbsHportion]
                # field_names = ['HExpPortion']
                # field_values = [H_exp_portion]
                # folder = "AC2D" + "/plots_struct/"
                # os.makedirs(folder, exist_ok=True)
                # for t in range(Qn.shape[-1]):
                    # print(f"Time Step = {t}")

                    # print(f"Qn/Hn = {Qn[0, :, :, t].abs().sum() / Hn[0, :, :, t].abs().sum()}")
                    # print(f"Qn/(Hn+Qn) = {Qn[0, :, :, t].abs().sum() / (Hn[0, :, :, t]+Qn[0, :, :, t]).abs().sum()}")
                    # print(f"Hn/(Hn+Qn) = {Hn[0, :, :, t].abs().sum() / (Hn[0, :, :, t]+Qn[0, :, :, t]).abs().sum()}")
                    # for field_name, shot in zip(field_names, field_values):
                    #     interpolation_opt = 'lanczos'
                    #     plt.figure()
                    #     shot = shot[0, :, :, t].cpu().numpy()
                    #     plt.imshow(shot, extent=(-np.pi, np.pi, -np.pi, np.pi), origin='lower', cmap='jet', vmin=0.0, vmax=1.0,
                    #                aspect='equal', interpolation=interpolation_opt)

                    #     # plt.colorbar()
                    #     plt.axis('off')
                    #     # plt.title(f'{field_name} at T={time_step+1}')
                    #     time_step_formatted = str(t + 1).zfill(3)
                    #    plot_name = folder + f'T_{time_step_formatted}_{field_name}'
                    #    plt.savefig(plot_name + '.png', dpi=300, bbox_inches='tight')
                    #    if t % 20 == 0:
                    #        plt.show()
                    #    plt.close()

        return self._compute_statistics()

    def _compute_statistics(self):
        self.test_l2_set = torch.tensor(self.test_l2_set)
        test_l2_avg = torch.mean(self.test_l2_set)
        test_l2_std = torch.std(self.test_l2_set)
        test_l2_min, min_idx = torch.min(self.test_l2_set), torch.argmin(self.test_l2_set)
        test_l2_max, max_idx = torch.max(self.test_l2_set), torch.argmax(self.test_l2_set)
        test_l2_mode, mode_count = torch.mode(self.test_l2_set)
        mode_indices = torch.nonzero(self.test_l2_set == test_l2_mode).squeeze().tolist()

        print("The average testing error is", test_l2_avg.item())
        print("Std. deviation of testing error is", test_l2_std.item())
        print("Min testing error is", test_l2_min.item(), "at index", min_idx.item())
        print("Max testing error is", test_l2_max.item(), "at index", max_idx.item())
        print("Mode of testing errors is", test_l2_mode.item(), "appearing", mode_count.item(), "times at indices",
              mode_indices)

        return {
            "input": self.inp,
            "exact": self.exact,
            "prediction": self.pred,
            "average": test_l2_avg.item(),
            "std_dev": test_l2_std.item(),
            "min": {"value": test_l2_min.item(), "index": min_idx.item()},
            "max": {"value": test_l2_max.item(), "index": max_idx.item()},
            "mode": {"value": test_l2_mode.item(), "count": mode_count.item(), "indices": mode_indices}
        }
