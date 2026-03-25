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
from collections import defaultdict

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
        # print(f"Available keys in self.data: {list(self.data.keys())}")
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
    def __init__(self, parent_dir, matlab_dataset, normalized, T_in, T_out, nTrjTrain, nTrjTest,
                 use_sliding_window=False):
        self.y = None
        self.x = None
        self.T_in = T_in
        self.T_out = T_out
        self.nTrain = nTrjTrain
        self.nTest = nTrjTest
        self.normalized = normalized
        self.normalizer_x = None
        self.normalizer_y = None
        self.navier = True if "ns" in matlab_dataset else False
        self.use_sliding_window = use_sliding_window

        matlab_dataset = parent_dir + matlab_dataset
        base, _ = os.path.splitext(matlab_dataset)
        python_dataset = base + '.pt'
        os.makedirs(parent_dir, exist_ok=True)

        if os.path.exists(python_dataset):
            print("Found saved dataset at", python_dataset)
            self.data = torch.load(python_dataset)['data'][:nTrjTrain + nTrjTest, 10:]  # TODO: it can affect the result of trained networks
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
        elif self.use_sliding_window:
            seq_len = self.T_in + self.T_out
            total_time = self.data.shape[1]
            num_samples = total_time - seq_len + 1

            # Initialize lists to hold sequences
            x_list = []
            y_list = []

            for i in range(num_samples):
                x_seq = self.data[:, i:i + self.T_in, *[slice(None)] * (self.data.ndim - 3)]
                y_seq = self.data[:, i + self.T_in:i + seq_len, *[slice(None)] * (self.data.ndim - 3)]
                x_list.append(x_seq)
                y_list.append(y_seq)

            # 1) stack to get [num_windows, batch, T_in, ...]
            self.x = torch.stack(x_list, dim=0)
            self.y = torch.stack(y_list, dim=0)

            # 2) swap the first two dims → [batch, num_windows, T_in, ...]
            #    permute args: (1, 0, 2, 3, ...)
            self.x = self.x.permute(1, 0, *range(2, self.x.ndim))
            self.y = self.y.permute(1, 0, *range(2, self.y.ndim))

            # 3) now flatten the first two dims → [batch * num_windows, T_in, ...]
            self.x = self.x.flatten(0, 1)
            self.y = self.y.flatten(0, 1)

            # Permute to match the original code's dimension order
            permute_order = list(range(self.x.ndim))
            permute_order.append(permute_order.pop(1))  # Move time to end

            self.x = self.x.permute(*permute_order)
            self.y = self.y.permute(*permute_order)
        else:
            permute_order = list(range(self.data.ndim))
            permute_order.append(permute_order.pop(1))  # Move the second dimension to the end
            self.x = self.data[:, :self.T_in, *[slice(None)] * (self.data.ndim - 3)].permute(*permute_order)
            self.y = self.data[:, self.T_in:self.T_in + self.T_out, *[slice(None)] * (
                    self.data.ndim - 3)].permute(*permute_order)
        print('Dataset (Train + Test) input shape = ', self.x.shape)
        print('Dataset (Train + Test) output shape = ', self.y.shape)
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
    def __init__(self, model, full_dataset, test_dataset, s, T_in, T_out, device, normalized=False, normalizers=None,
                 time_history=False, use_sliding_window=False):
        self.model = model
        self.full_dataset = full_dataset
        self.test_dataset = test_dataset
        self.s, self.T_in, self.T_out = s, T_in, T_out
        self.device = device
        self.normalized = normalized
        if normalized:
            self.normalizer_x = normalizers[0].to(self.device)
            self.normalizer_y = normalizers[1].to(self.device)

        # compute how many windows per trajectory
        nTraj, total_time = full_dataset.data.shape[0], full_dataset.data.shape[1]
        self.windows_per_traj = total_time - (T_in + T_out) + 1
        self.total_time = total_time

        # recover the set of test-trajectory IDs
        self.test_traj_ids = {
            idx // self.windows_per_traj
            for idx in test_dataset.indices
        }

        self.time_history = time_history
        self.use_sliding_window = use_sliding_window
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Buffers for the windows-based test set (if you still need them)
        n_test_traj = len(self.test_traj_ids)
        spatial_dims = list(self.full_dataset.data.shape[2:])
        T_total = self.full_dataset.data.shape[1] - T_in
        self.inp = torch.zeros((n_test_traj, *spatial_dims, T_in), device=self.device)
        self.exact = torch.zeros((n_test_traj, *spatial_dims, T_total), device=self.device)
        self.pred = torch.zeros((n_test_traj, *spatial_dims, T_total), device=self.device)

        # overall trajectory L2s
        self.test_l2_set = []
        # per‐time‐step L2s: time_idx → list of errors
        self.window_errors = defaultdict(list)

    def evaluate(self, loss_fn):
        self.model.eval()
        self.test_l2_set.clear()
        self.window_errors.clear()

        if self.use_sliding_window:
            # bring time to last dim
            permute_order = list(range(self.full_dataset.data.ndim))
            permute_order.append(permute_order.pop(1))
            data_perm = self.full_dataset.data.permute(*permute_order)

            with torch.no_grad():
                index = 0
                for traj_id in sorted(self.test_traj_ids):
                    # ground truth for this trajectory: [1, ... , T]
                    true = data_perm[traj_id].unsqueeze(0).to(self.device)

                    # prediction buffer
                    pred = torch.zeros_like(true, device=self.device)
                    # seed the first T_in from ground truth
                    pred[..., :self.T_in] = true[..., :self.T_in]

                    start_time = time.time()
                    t = 0
                    # we no longer need a separate 'window_idx' for blocks; errors are stored per absolute time
                    while t + self.T_in < self.total_time:
                        # prepare input window of length T_in
                        x_in = pred[..., t: t + self.T_in]
                        if self.normalized:
                            x_in = self.normalizer_x.encode(x_in)

                        # model predicts next T_out frames all at once
                        if self.time_history:
                            for tp in range(0, self.T_out):
                                im = self.model(x_in)
                                if tp == 0:
                                    out = im
                                else:
                                    out = torch.cat((out, im), -1)
                                x_in = im
                            if self.normalized:
                                out = self.normalizer_y.decode(out)

                        else:
                            out = self.model(x_in)
                            if self.normalized:
                                out = self.normalizer_y.decode(out)

                        # write the predicted block into the rolling buffer
                        start = t + self.T_in
                        end = start + self.T_out
                        pred[..., start:end] = out

                        # compute per‐time‐step L2 within this block ──
                        # ‘out’ has shape [..., T_out], and true[..., start:end] is the ground‐truth block.
                        for dt in range(self.T_out):
                            pred_t = out[..., dt: dt + 1]  # shape: [..., 1]
                            true_t = true[..., start + dt: start + dt + 1]  # shape: [..., 1]
                            one_step_l2 = loss_fn(
                                pred_t.reshape(1, -1),
                                true_t.reshape(1, -1)
                            ).item()
                            absolute_time = start + dt  # use the absolute time‐index as the “window key”
                            self.window_errors[absolute_time].append(one_step_l2)

                        # advance by T_out in the rollout
                        t += self.T_out

                    # after rolling out the full trajectory, compute its overall L2 (excluding t=0 if desired)
                    traj_l2 = loss_fn(
                        pred[..., 1:].reshape(1, -1),
                        true[..., 1:].reshape(1, -1)
                    ).item()

                    # print(f"time of predicting trajectory = {time.time() - start_time} - Trajectory ID = {index}, L2 error = {traj_l2}")

                    self.test_l2_set.append(traj_l2)
                    # print(f"traj {traj_id:2d}  traj-L2 = {traj_l2:.5f}")

                    self.inp[index] = true.squeeze(0)[..., :self.T_in]  # initial condition
                    self.exact[index] = true.squeeze(0)[..., self.T_in:]  # entire ground truth
                    self.pred[index] = pred.squeeze(0)[..., self.T_in:]  # entire model rollout
                    index += 1

        elif self.time_history:
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
                        dt = time.time() - t_start
                        y = self.normalizer_y.decode(y)
                        x = self.normalizer_x.decode(x)
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
        # overall trajectory stats (with quartiles)
        all_traj = torch.tensor(self.test_l2_set, dtype=torch.float32)
        q1_traj = torch.quantile(all_traj, 0.25).item()
        q2_traj = torch.quantile(all_traj, 0.50).item()  # median
        q3_traj = torch.quantile(all_traj, 0.75).item()

        all_traj_mode, mode_count = torch.mode(all_traj)
        mode_indices = torch.nonzero(all_traj == all_traj_mode).squeeze().tolist()

        stats = {
            "trajectory": {
                "count": len(all_traj),
                "average": all_traj.mean().item(),
                "std_dev": all_traj.std().item(),
                "min": {"value": all_traj.min().item(), "index": all_traj.argmin().item()},
                "max": {"value": all_traj.max().item(), "index": all_traj.argmax().item()},
                "mode": {"value": all_traj_mode.item(), "count": mode_count.item(), "indices": mode_indices},
                "q1": q1_traj,
                "median": q2_traj,
                "q3": q3_traj
            }
        }

        # per‐time‐step stats (including quartiles)
        window_stats = {}
        for time_idx, errors in self.window_errors.items():
            errs = torch.tensor(errors, dtype=torch.float32)

            q1 = torch.quantile(errs, 0.25).item()
            q2 = torch.quantile(errs, 0.50).item()
            q3 = torch.quantile(errs, 0.75).item()

            window_stats[time_idx] = {
                "count": len(errs),
                "average": errs.mean().item(),
                "std_dev": errs.std().item(),
                "min": errs.min().item(),
                "max": errs.max().item(),
                "q1": q1,
                "median": q2,
                "q3": q3
            }

        stats["per_window"] = window_stats

        return {
            "input": self.inp,
            "exact": self.exact,
            "prediction": self.pred,
            **stats
        }
