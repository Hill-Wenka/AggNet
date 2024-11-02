import torch
import torch.nn as nn


# maximum mean discrepancy (MMD) loss for Wasserstein autoencoder (WAE)
class MMDLoss(nn.Module):
    def __init__(self, method='full_kernel', kernel='Gaussian', sigma=1.0, rf_dim=500, **kwargs):
        super(MMDLoss, self).__init__()

        self.method = method  # 'full_kernel' or 'random_feature'
        self.kernel = kernel
        self.sigma = sigma
        self.rf_dim = rf_dim  # random feature dimension
        self.reduction = kwargs.get('reduction', 'mean')

    def Gaussian_kernel(self, x, y):
        # exp(-||x-y||_2^2 / (2*sigma^2)), RBF is the expnential of the minus squared Euclidean distance
        tiled_x = x.unsqueeze(1)  # (N, 1, D)
        tiled_y = y.unsqueeze(0)  # (1, N, D)
        pairwise_distances = ((tiled_x - tiled_y) ** 2).sum(dim=2)  # (N, N, D) -> (N, N)
        return torch.exp(-pairwise_distances / (2 * self.sigma ** 2))  # (N, N)

    def compute_kernel(self, x, y, diag=True):
        if self.kernel == 'Gaussian':
            pairwise_distances = self.Gaussian_kernel(x, y)  # (N, N)
        else:
            raise ValueError('Invalid kernel type')

        if not diag:  # remove the diagonal elements
            pairwise_distances = pairwise_distances - torch.diag(torch.diag(pairwise_distances))  # (N, N) -> (N, N)
        return pairwise_distances

    def compute_full_kernel_mmd(self, x, y):
        N = x.shape[0]
        K_xx = self.compute_kernel(x, x, diag=False).sum() / (N / (N - 1))  # (N, N) -> scalar
        K_yy = self.compute_kernel(y, y, diag=False).sum() / (N / (N - 1))  # (N, N) -> scalar
        K_xy = self.compute_kernel(x, y, diag=True).sum() / N ** 2  # (N, N) -> scalar
        mmd = K_xx + K_yy - 2 * K_xy  # scalar
        return mmd

    def compute_rf_kernel(self, z, rf_w, rf_b):
        if self.kernel == 'Gaussian':
            z_emb = (z @ rf_w) / self.sigma + rf_b  # (N, d)
            z_emb = torch.cos(z_emb) * (2. / self.rf_dim) ** 0.5  # (N, d)
        else:
            raise ValueError('Invalid kernel type')
        mu_rf = z_emb.mean(0, keepdim=False)
        return mu_rf

    def compute_random_feature_mmd(self, x, y):
        N, D = x.shape
        rf_w = torch.randn(D, self.rf_dim)  # (D, d)
        rf_b = 2 * math.pi * torch.rand((rf_dim,))  # (d,)
        mu_x = self.compute_rf_kernel(x, rf_w, rf_b)  # (N, d)
        mu_y = self.compute_rf_kernel(y, rf_w, rf_b)  # (N, d)
        mmd = ((mu_x - mu_y) ** 2).sum()  # scalar
        return mmd

    def forward(self, z):
        z_prior = torch.randn_like(z)  # (N, D)
        if self.method == 'full_kernel':
            mmd_loss = self.compute_full_kernel_mmd(z, z_prior)
        elif self.method == 'random_feature':
            mmd_loss = self.compute_random_feature_mmd(z, z_prior)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            pass
        elif self.reduction == 'sum':
            mmd_loss = mmd_loss * z.shape[0]
        else:
            raise ValueError('Invalid reduction type')
        return mmd_loss
