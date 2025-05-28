from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from diffusion_utils import EDMLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses import *
import os 

ModuleType = Union[str, Callable[..., nn.Module]]

randn_like=torch.randn_like

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t = 512):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, noise_labels, class_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)

        x = self.proj(x) 
        x = x + emb
        return self.mlp(x)


class Precond(nn.Module):
    def __init__(self,
        denoise_fn,
        hid_dim,
        sigma_min = 0,                # Minimum supported noise level.
        sigma_max = float('inf'),     # Maximum supported noise level.
        sigma_data = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        ###########
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma):

        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    

class Model(nn.Module):
    def __init__(self, denoise_fn, hid_dim, P_mean=-1.2, P_std=1.2, sigma_data=0.5, gamma=5, opts=None, pfgmpp = False):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

    def forward(self, x):

        loss = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean()

class DiffPuter(nn.Module):

    def __init__(self, train_X: np.array, test_X: np.array, train_mask: np.array, test_mask: np.array,
                result_save_path: str = "../",
                 num_trials: int = 10, epochs_m_step: int = 10000, patience_m_step: int = 300, batch_size: int = 4096,
                 hid_dim: int = 1024, device: str = "cuda", max_iter: int = 10, lr: float = 1e-4, num_steps: int = 50, ckpt_dir: str = "ckpt"):

        # parameters for the whole training step
        self.max_iter = max_iter
        self.ckpt_dir = ckpt_dir
        self.result_save_path = result_save_path

        # parameters for M step
        self.epochs_m_step = epochs_m_step
        self.batch_size = batch_size
        self.patience_m_step = patience_m_step
        self.hid_dim = hid_dim
        self.lr = lr
        self.in_dim = train_X.shape[1]
        self.device = device
        
        # parameters for E step
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.SIGMA_MIN=0.002
        self.SIGMA_MAX=80
        self.rho=7
        self.S_churn= 1
        self.S_min=0
        self.S_max=float('inf')
        self.S_noise=1

        # data
        self.mean_X = train_X.mean(0)
        self.std_X = train_X.std(0)

        self.X = self.standardize_data(train_X)

        self.X_test = self.standardize_data(test_X)

        self.mask_train = torch.Tensor(train_mask)
        self.mask_test = torch.Tensor(test_mask)

    def compute_metrics(self, iteration, rec_Xs, X_true, mask):

        rec_X = torch.stack(rec_Xs, dim = 0).mean(0) 
        print("rex_X_computemetrics", rec_X)
        rec_X = rec_X.cpu().numpy() * 2
        X_true = X_true.cpu().numpy() * 2

        print("X_true_computemetrics", X_true)
        np.save(f'{self.ckpt_dir}/iter_{iteration+1}.npy', rec_X)

        pred_X = rec_X[:]
        
        pred_X = rec_X * self.std_X + self.mean_X
        

        print("pred_X_computeMetrics", pred_X)
        mae, rmse= self.get_eval(pred_X, X_true, mask)

        return mae, rmse

    def fit(self):

        for iteration in range(self.max_iter):

            print('"iteração:', iteration)
            os.makedirs(f'{self.ckpt_dir}/{iteration}', exist_ok=True) if not os.path.exists(f'{self.ckpt_dir}/{iteration}') else None

            self._M_step(iteration)

            # reconstructed training data during the E-step 
            rec_Xs_train = self._E_Step(iteration, self.X, self.mask_train)

            print("recxstrain",rec_Xs_train[0])

            mae_train, rmse_train = self.compute_metrics(iteration, rec_Xs_train, self.X, self.mask_train)
        
            # reconstructed test data during the E-step
            rec_Xs_test = self._E_Step(iteration, self.X_test, self.mask_test)

            print("recxstest",rec_Xs_test[0])

            mae_test, rmse_test = self.compute_metrics(iteration, rec_Xs_test, self.X_test, self.mask_test)

            with open (f'{self.result_save_path}/result.txt', 'a+') as f:

                f.write(f'iteration {iteration}, MAE: in-sample: {mae_train}, out-of-sample: {mae_test} \n')
                f.write(f'iteration {iteration}: RMSE: in-sample: {rmse_train}, out-of-sample: {rmse_test} \n')

            print('in-sample', mae_train, rmse_train)
            print('out-of-sample', mae_test, rmse_test)

            print(f'saving results to {self.result_save_path}')

    def standardize_data(self, X: np.array) -> np.array:

        X_stdized = (X - self.mean_X) / self.std_X / 2
        X_stdized = torch.tensor(X_stdized)

        return X_stdized
    
    def _M_step(self, iteration):
        if iteration == 0:
            X_miss = (1. - self.mask_train.float()) * self.X
            train_data = X_miss.numpy()
        else:
            print(f'Loading X_miss from {self.ckpt_dir}/iter_{iteration}.npy')
            X_miss = np.load(f'{self.ckpt_dir}/iter_{iteration}.npy') / 2
            train_data = X_miss
 
        train_loader = DataLoader(
            train_data,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 4,
        )

        print("X miss M step:", X_miss)
        denoise_fn = MLPDiffusion(self.in_dim, self.hid_dim).to(self.device)
        print(denoise_fn)

        num_params = sum(p.numel() for p in denoise_fn.parameters())
        print("the number of parameters", num_params)

        model = Model(denoise_fn = denoise_fn, hid_dim = self.in_dim).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=40, verbose=True)

        model.train()

        best_loss = float('inf')
        patience = 0
        for epoch in range(self.epochs_m_step):

            batch_loss = 0.0
            len_input = 0
            for batch in train_loader:
                inputs = batch.float().to(self.device)
                loss = model(inputs)

                loss = loss.mean()
                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            curr_loss = batch_loss/len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                torch.save(model.state_dict(), f'{self.ckpt_dir}/{iteration}/model.pt')
            else:
                patience += 1
                if patience == self.patience_m_step:
                    print('Early stopping')
                    break
            print(f'Epoch {epoch+1}/{self.epochs_m_step}, Loss: {curr_loss:.4f}, Best loss: {best_loss:.4f}')

            if epoch % 1000 == 0:
                torch.save(model.state_dict(), f'{self.ckpt_dir}/{iteration}/model_{epoch}.pt')

        return 0

    def _E_Step(self, iteration, X, mask)-> List:

        rec_Xs = []

        print("X E step", X)

        for trial in range(self.num_trials):
        
            X_miss = (1. - mask.float()) * X
            X_miss = X_miss.to(self.device)
            impute_X = X_miss

            # print("X_Miss", X_miss)
            # print("X", X)
  
            in_dim = X.shape[1]

            denoise_fn = MLPDiffusion(in_dim, self.hid_dim).to(self.device) #self.in_dim

            model = Model(denoise_fn = denoise_fn, hid_dim = in_dim).to(self.device)
            model.load_state_dict(torch.load(f'{self.ckpt_dir}/{iteration}/model.pt'))

            # ==========================================================

            net = model.denoise_fn_D

            num_samples, dim = X.shape[0], X.shape[1]

            print("impute_X", impute_X)
            rec_X = self.impute_mask(net, impute_X, mask, num_samples, dim)

            print("rec_X", rec_X)
            
            mask_int = mask.to(torch.float).to(self.device)
            rec_X = rec_X * mask_int + impute_X * (1-mask_int)
            rec_Xs.append(rec_X)
            
            print("rec_X after", rec_X)
            print(f'Trial = {trial}')

        return rec_Xs
    
    def impute_mask(self, net, x, mask, num_samples, dim):
        step_indices = torch.arange(self.num_steps, dtype=torch.float32, device=self.device)
        x_t = torch.randn([num_samples, dim], device=self.device)

        sigma_min = max(self.SIGMA_MIN, net.sigma_min)
        sigma_max = min(self.SIGMA_MAX, net.sigma_max)

        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        mask = mask.to(torch.int).to(self.device)
        x_t = x_t.to(torch.float32) * t_steps[0]

        N = 10
        with torch.no_grad():

            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                if i < self.num_steps - 1:
            
                    for j in range(N):
                        n_curr = torch.randn_like(x_t).to(self.device) * t_cur
                        n_prev = torch.randn_like(x_t).to(self.device) * t_next

                        x_known_t_prev = x + n_prev
                        x_unknown_t_prev = self.sample_step(net, i, t_cur, t_next, x_t)

                        x_t_prev = x_known_t_prev * (1-mask) + x_unknown_t_prev * mask

                        n = torch.randn_like(x_t) * (t_cur.pow(2) - t_next.pow(2)).sqrt()

                        if j == N - 1:
                            x_t = x_t_prev                                                # turn to x_{t-1}
                        else:
                            x_t = x_t_prev + n                                            # new x_t

        return x_t

    def sample_step(self, net, i, t_cur, t_next, x_next):

        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur) 
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * randn_like(x_cur)
        # Euler step.

        denoised = net(x_hat, t_hat).to(torch.float32)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < self.num_steps - 1:
            denoised = net(x_next, t_next).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next
    
    def get_eval(self, X_recon: np.array, X_true: np.array, mask: np.array) -> Tuple[float, float]:

        # mae = np.abs(num_pred[num_mask] - num_true[num_mask]).mean()
        # rmse = np.sqrt(((num_pred[num_mask] - num_true[num_mask])**2).mean())
        print("mask", mask)
        print("X_recon", X_recon)
        mask = mask.numpy()
        mask = mask.astype(bool)

        mae = np.nanmean(np.abs(X_recon[mask]- X_true[mask]))
        rmse = np.sqrt(np.nanmean((X_recon[mask]- X_true[mask])**2))

        return mae, rmse