import os 
import gc
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

epsilon = 1e-8

class Bayesian_SpatialCorr(nn.Module):
    """
    Bayesian Spatial Correction Loss Module

    Implements a loss function that takes into account the covariance structure of both 
    the prior and posterior distributions for label error η in a segmentation task.

    Args:
        model (nn.Module): Segmentation model.
        class_num (int): Number of segmentation classes. Default is 2.
        rhosigma_init (float): Initial value for the correlation coefficient in the prior distribution. Default is 0.5.
        rhogamma_init (float): Initial value for the correlation coefficient in the posterior distribution. Default is 0.5.
        mu_init (float): Initial mean (μ) for the prior distribution. Default is -1.5.
        sigma_init (float): Initial σ for the prior distribution. Default is 0.1.
        m_init (float): Initial mean (m) for the posterior distribution. Default is -1.5.
        gamma_init (float): Initial γ for the posterior distribution. Default is 0.1.
        imsize (int): Image size (assuming square images). Default is 512.
        trainable (bool): Whether the parameters are trainable. Default is False.
        estep (int): Number of iterations in the E-step. Default is 1.

    Note:
        The computation of the base terms in the forward method now uses the internal _base_terms method.
    """
    def __init__(
        self,
        model: nn.Module,
        class_num: int = 2,
        rhosigma_init: float = 0.5,
        rhogamma_init: float = 0.5,
        mu_init: float = -1.5,
        sigma_init: float = 0.1,
        m_init: float = -1.5,
        gamma_init: float = 0.1,
        imsize: int = 512,
        trainable: bool = False,
        estep: int = 1,
    ):
        super(Bayesian_SpatialCorr, self).__init__()
        self.model = model
        self.class_num = class_num
        self.imsize = imsize
        self.m_init = m_init
        self.gamma_init = gamma_init
        self.trainable = trainable
        self.estep = estep

        # Parameters for the prior distribution of η
        self.rho_sigma = nn.Parameter(torch.tensor(rhosigma_init, dtype=torch.float32), requires_grad=trainable)
        self.mu = nn.Parameter(torch.ones(imsize, imsize, dtype=torch.float32) * mu_init, requires_grad=trainable)
        self.sigma = nn.Parameter(torch.tensor([sigma_init], dtype=torch.float32), requires_grad=trainable)

        # Parameter for the posterior distribution of η (correlation coefficient)
        self.rho_gamma = nn.Parameter(torch.tensor(rhogamma_init, dtype=torch.float32), requires_grad=trainable)


    def add_variable(self, cfg):
        """
        Set additional variables (e.g., save paths) from the configuration.
        """
        self.cfg = cfg

    def _set_postparams_worker(self, args):
        """
        Initialize and save the posterior distribution parameters (m, Γ) for each image.

        Args:
            args (tuple): (cfg, filename, imsize)
        """
        cfg, filename, imsize = args
        file_id = filename.split('.')[0].replace("_label", "")
        m = np.ones((imsize, imsize), dtype=np.float32) * cfg.loss.m_init
        G = np.ones((imsize, imsize), dtype=np.float32) * cfg.loss.gamma_init
        param = np.stack([m, G])
        param_path = os.path.join(cfg.loss.params_base_storage, f'{file_id}.npy')
        np.save(param_path, param)

    def _resave_postparams_worker(self, args):
        """
        Resave the optimized posterior parameters after processing.

        Args:
            args (tuple): (cfg, q_param, file_id, flip_horizontal, flip_vertical, save2nfs)
        """
        cfg, q_param, file_id, flip_horizontal, flip_vertical, save2nfs = args

        # Reverse flip transformation if applied
        if flip_horizontal.item():
            q_param = torchvision.transforms.functional.hflip(q_param)
        if flip_vertical.item():
            q_param = torchvision.transforms.functional.vflip(q_param)

        param = q_param.detach().clone().cpu().numpy()
        param_path = os.path.join(cfg.loss.params_base_storage, f'{file_id}.npy')
        np.save(param_path, param)
        if save2nfs:
            post_dist_savepath = f"{cfg.utils.save_dir}/tmp_postdists/{file_id}.npy"
            np.save(post_dist_savepath, param)

    def set_postparams(self, cfg, train_filenames):
        """
        Set and save the initial posterior parameters (m, Γ) for each training image.

        Args:
            cfg: Configuration object.
            train_filenames (list): List of training image filenames.
        """
        print('Setting initial posterior parameters (m, Γ) for η...')
        print(f'Saving parameters to {cfg.loss.params_base_storage}')
        for filename in tqdm(train_filenames, total=len(train_filenames)):
            self._set_postparams_worker((cfg, filename, self.imsize))

    def resave_postparams(self, cfg, params, train_fileids, flip_horizontals, flip_verticals, save2nfs=False):
        """
        Resave the optimized posterior parameters (m, Γ).

        Args:
            cfg: Configuration object.
            params (Tensor): Optimized parameters.
            train_fileids (list): List of image IDs.
            flip_horizontals (list): List of horizontal flip flags.
            flip_verticals (list): List of vertical flip flags.
            save2nfs (bool): Whether to additionally save to NFS.
        """
        for file_id, param, flip_horizontal, flip_vertical in zip(train_fileids, params, flip_horizontals, flip_verticals):
            self._resave_postparams_worker((cfg, param, file_id, flip_horizontal, flip_vertical, save2nfs))
        del params
        gc.collect()
        torch.cuda.empty_cache()

    def _log_detSigma(self):
        """
        Compute the log-determinant of the prior covariance matrix Σ.
        """
        sigma = F.softplus(self.sigma)  # Ensure sigma is positive
        sigma_params = sigma.expand(self.imsize * self.imsize).clone() + epsilon
        log_detSigma = torch.log(sigma_params).sum() * 2 + \
                       self.imsize * (self.imsize - 1) * torch.log(1 - self.rho_sigma**2) * 2
        return log_detSigma

    def _log_detGamma(self, G):
        """
        Compute the log-determinant of the posterior covariance matrix Γ.

        Args:
            G (Tensor): Tensor with shape [B, H, W].
        """
        log_detGamma = torch.log(G + epsilon).sum([1, 2]) * 2 + \
                       self.imsize * (self.imsize - 1) * torch.log(1 - self.rho_gamma**2) * 2
        return log_detGamma

    def _createtensor_Sigma(self, rho, location: str):
        """
        Generate a tensor used for the Kronecker product of Σ based on grid location.

        Args:
            rho (float): Correlation coefficient.
            location (str): One of 'centor', 'upper', 'bottom', 'left', 'right', 
                            'up_left', 'up_right', 'bottom_left', 'bottom_right'.
        Returns:
            Tensor: Partial tensor representing covariance structure.
        """
        match location:
            case 'centor':
                return torch.tensor([
                    rho**2, -rho*(1+rho**2), rho**2,
                    -rho*(1+rho**2), (1+rho**2)**2, -rho*(1+rho**2),
                    rho**2, -rho*(1+rho**2), rho**2
                ], dtype=torch.float32)
            case 'upper':
                return torch.tensor([
                    0, 0, 0, -rho, 1+rho**2, -rho, rho**2, -rho*(1+rho**2), rho**2
                ], dtype=torch.float32)
            case 'bottom':
                return torch.tensor([
                    rho**2, -rho*(1+rho**2), rho**2, -rho, 1+rho**2, -rho, 0, 0, 0
                ], dtype=torch.float32)
            case 'left':
                return torch.tensor([
                    0, -rho, rho**2, 0, 1+rho**2, -rho*(1+rho**2), 0, -rho, rho**2
                ], dtype=torch.float32)
            case 'right':
                return torch.tensor([
                    rho**2, -rho, 0, -rho*(1+rho**2), 1+rho**2, 0, rho**2, -rho, 0 
                ], dtype=torch.float32)
            case 'up_left':
                return torch.tensor([
                    0, 0, 0, 0, 1, -rho, 0, -rho, rho**2
                ], dtype=torch.float32)
            case 'up_right':
                return torch.tensor([
                    0, 0, 0, -rho, 1, 0, rho**2, -rho, 0
                ], dtype=torch.float32)
            case 'bottom_left':
                return torch.tensor([
                    0, -rho, rho**2, 0, 1, -rho, 0, 0, 0
                ], dtype=torch.float32)
            case 'bottom_right':
                return torch.tensor([
                    rho**2, -rho, 0, -rho, 1, 0, 0, 0, 0
                ], dtype=torch.float32)
            case _:
                raise ValueError(f"Invalid location: {location}")

    def _createtensor_Gamma(self, rho, location: str):
        """
        Generate a tensor used for the Kronecker product of Γ based on grid location.

        Args:
            rho (float): Correlation coefficient.
            location (str): One of 'centor', 'upper', 'bottom', 'left', 'right', 
                            'up_left', 'up_right', 'bottom_left', 'bottom_right'.
        Returns:
            Tensor: Partial tensor representing covariance structure.
        """
        match location:
            case 'centor':
                return torch.tensor([
                    rho**2, rho, rho**2,
                    rho, 1, rho,
                    rho**2, rho, rho**2
                ], dtype=torch.float32)
            case 'upper':
                return torch.tensor([
                    0, 0, 0, rho, 1, rho, rho**2, rho, rho**2
                ], dtype=torch.float32)
            case 'bottom':
                return torch.tensor([
                    rho**2, rho, rho**2, rho, 1, rho, 0, 0, 0
                ], dtype=torch.float32)
            case 'left':
                return torch.tensor([
                    0, rho, rho**2, 0, 1, rho, 0, rho, rho**2
                ], dtype=torch.float32)
            case 'right':
                return torch.tensor([
                    rho**2, rho, 0, rho, 1, 0, rho**2, rho, 0
                ], dtype=torch.float32)
            case 'up_left':
                return torch.tensor([
                    0, 0, 0, 0, 1, rho, 0, rho, rho**2
                ], dtype=torch.float32)
            case 'up_right':
                return torch.tensor([
                    0, 0, 0, rho, 1, 0, rho**2, rho, 0
                ], dtype=torch.float32)
            case 'bottom_left':
                return torch.tensor([
                    0, rho, rho**2, 0, 1, rho, 0, 0, 0
                ], dtype=torch.float32)
            case 'bottom_right':
                return torch.tensor([
                    rho**2, rho, 0, rho, 1, 0, 0, 0, 0
                ], dtype=torch.float32)
            case _:
                raise ValueError(f"Invalid location: {location}")

    def _set_paramindices(self):
        """
        Define the indices for different regions in the image grid (upper, lower, left, right, corners).

        Returns:
            Tuple containing indices for various grid regions.
        """
        indices_upper = range(1, self.imsize - 1)
        indices_bottom = range(self.imsize * (self.imsize - 1) + 1, self.imsize ** 2 - 1)
        indices_left = range(self.imsize, self.imsize * (self.imsize - 1), self.imsize)
        indices_right = range(2 * self.imsize - 1, self.imsize ** 2 - 1, self.imsize)
        indices_upleft = [0]
        indices_upright = [self.imsize - 1]
        indices_bottomleft = [self.imsize * (self.imsize - 1)]
        indices_bottomright = [self.imsize ** 2 - 1]
        return indices_upper, indices_bottom, indices_left, indices_right, indices_upleft, indices_upright, indices_bottomleft, indices_bottomright

    def _invSigma(self, N):
        """
        Compute the inverse covariance matrix Σ for the prior distribution given batch size N.

        Args:
            N (int): Batch size.
        Returns:
            Tensor with shape [N, H*W, H*W] representing the inverse covariance matrix.
        """
        indices_upper, indices_bottom, indices_left, indices_right, indices_upleft, indices_upright, indices_bottomleft, indices_bottomright = self._set_paramindices()
        # Create base tensor using the 'centor' region
        kron_Ry_Rx = torch.tile(self._createtensor_Sigma(self.rho_sigma, 'centor'), (self.imsize * self.imsize, 1)).cuda()

        def set_values(indices, values):
            for idx in indices:
                kron_Ry_Rx[idx] = values

        set_values(indices_upper, self._createtensor_Sigma(self.rho_sigma, 'upper'))
        set_values(indices_bottom, self._createtensor_Sigma(self.rho_sigma, 'bottom'))
        set_values(indices_left, self._createtensor_Sigma(self.rho_sigma, 'left'))
        set_values(indices_right, self._createtensor_Sigma(self.rho_sigma, 'right'))
        set_values(indices_upleft, self._createtensor_Sigma(self.rho_sigma, 'up_left'))
        set_values(indices_upright, self._createtensor_Sigma(self.rho_sigma, 'up_right'))
        set_values(indices_bottomleft, self._createtensor_Sigma(self.rho_sigma, 'bottom_left'))
        set_values(indices_bottomright, self._createtensor_Sigma(self.rho_sigma, 'bottom_right'))

        sigma = F.softplus(self.sigma)
        inv_Sigma = (kron_Ry_Rx * (1 / sigma) * (1 / sigma)) / ((1 - self.rho_sigma**2) ** 2)
        inv_Sigma = inv_Sigma.unsqueeze(0).expand(N, -1, -1).cuda()
        return inv_Sigma

    def _expand_G(self, G):
        """
        Pad and expand the tensor G so that local information can be processed via convolution.

        Args:
            G (Tensor): Tensor with shape [N, H, W].
        Returns:
            Tensor: Expanded tensor with shape [N, 9, H*W].
        """
        N, H, W = G.shape
        padded_grid = F.pad(G.unsqueeze(1), (1, 1, 1, 1), mode='constant', value=0)
        unfold = F.unfold(padded_grid, kernel_size=(H, W))
        return unfold

    def _Gamma(self, N, G):
        """
        Compute the posterior covariance matrix Γ using the Kronecker product.

        Args:
            N (int): Batch size.
            G (Tensor): Tensor with shape [N, H, W] representing posterior parameters.
        Returns:
            Tensor: Γ matrix.
        """
        indices_upper, indices_bottom, indices_left, indices_right, indices_upleft, indices_upright, indices_bottomleft, indices_bottomright = self._set_paramindices()
        kron_Ry_Rx = torch.tile(self._createtensor_Gamma(self.rho_gamma, 'centor'), (self.imsize * self.imsize, 1)).cuda()

        def set_values(indices, values):
            for idx in indices:
                kron_Ry_Rx[idx] = values

        set_values(indices_upper, self._createtensor_Gamma(self.rho_gamma, 'upper'))
        set_values(indices_bottom, self._createtensor_Gamma(self.rho_gamma, 'bottom'))
        set_values(indices_left, self._createtensor_Gamma(self.rho_gamma, 'left'))
        set_values(indices_right, self._createtensor_Gamma(self.rho_gamma, 'right'))
        set_values(indices_upleft, self._createtensor_Gamma(self.rho_gamma, 'up_left'))
        set_values(indices_upright, self._createtensor_Gamma(self.rho_gamma, 'up_right'))
        set_values(indices_bottomleft, self._createtensor_Gamma(self.rho_gamma, 'bottom_left'))
        set_values(indices_bottomright, self._createtensor_Gamma(self.rho_gamma, 'bottom_right'))

        flatten_G = G.view(N, -1)
        unfold_G = self._expand_G(G)
        Gamma = flatten_G.unsqueeze(2) * kron_Ry_Rx * unfold_G
        return Gamma

    def _extend_deviation(self, dev):
        """
        Extract local deviation information by padding and unfolding the tensor.

        Args:
            dev (Tensor): Tensor with shape [N, H, W] representing deviations.
        Returns:
            Tensor: Unfolded tensor with shape [N*H*W, 9] representing 3x3 neighborhood.
        """
        N, H, W = dev.shape
        padded_grid = F.pad(dev.unsqueeze(1), (1, 1, 1, 1), mode='constant', value=0)
        unfold = F.unfold(padded_grid, kernel_size=(3, 3))
        unfold = unfold.permute(0, 2, 1).reshape(N * H * W, 9)
        return unfold

    def _compute_trace(self, dev, inv_Sigma):
        """
        Compute the trace term used in the loss.

        Args:
            dev (Tensor): Tensor with shape [N, H, W] representing deviation.
            inv_Sigma (Tensor): Inverse covariance matrix with shape [N, H*W, H*W].
        Returns:
            Tensor: Computed trace value for each batch.
        """
        N, H, W = dev.shape
        flatten_dev = dev.view(N, -1)
        expanded_dev = self._extend_deviation(dev)
        B_mu_minus_m = flatten_dev.unsqueeze(2) * inv_Sigma * expanded_dev.view(N, H * W, 9)
        return B_mu_minus_m.sum(dim=(1, 2))

    def _second_term(self, m, G):
        """
        Compute the second term in the loss, which relates the prior and posterior covariances.

        Args:
            m (Tensor): Posterior mean parameter with shape [B, H, W].
            G (Tensor): Posterior variance parameter with shape [B, H, W].
        Returns:
            Tensor: Loss value for the second term.
        """
        N, H, W = m.shape
        inv_Sigma = self._invSigma(N)
        Tr_g = self._compute_trace(m - self.mu.unsqueeze(0).expand(N, -1, -1), inv_Sigma)
        Gamma = self._Gamma(N, G)
        TrGSi = torch.sum(Gamma * inv_Sigma, dim=(1, 2))
        log_detSigma = self._log_detSigma()
        loss2 = - (Tr_g + TrGSi + (H * W * math.log(math.pi * 2)) + log_detSigma) / 2
        loss2 = loss2 / (self.imsize * self.imsize)
        return loss2.mean()

    def _base_terms(self, predictions, targets, m, G, image_id, epoch=None):
        """
        Compute the base terms (e.g., log-likelihood) using the reparameterization trick.

        Args:
            predictions (Tensor): (N, C, H, W) model predictions.
            targets (Tensor): (N, H, W) ground-truth labels.
            m (Tensor): (N, H, W) posterior mean parameter.
            G (Tensor): (N, H, W) posterior variance parameter.
            image_id (list): List of image IDs.
            epoch (int, optional): Current epoch number.

        Returns:
            Tuple[Tensor, Tensor]: (q_yt_log_p_yt, q_yt)

        The following commented-out code shows an alternative derivation (including term3 and term4)
        for reference:
        
            # total_loss = - (term1 + term2 - term3 + term4 + term5)
            # For binary segmentation, term3 and term4 may cancel out or be set to zero.
        """
        N, C, H, W = predictions.shape

        # Reparameterization trick for η: sample etas ~ m + sqrt(G) * epsilon
        etas = m + torch.randn_like(m) * torch.sqrt(G)  # [N, H, W]
        rs = torch.sigmoid(etas)  # [N, H, W]

        # Abbreviation: yt = y_true, yo = y_observed
        log_p_yt = torch.log(predictions + epsilon)  # [N, C, H, W]

        # For later debug/visualization: initialize q_yt and its log-likelihood
        q_yt = torch.zeros_like(predictions, requires_grad=False)  # [N, C, H, W]
        q_yt_log_p_yt = torch.zeros_like(etas)  # [N, H, W]
        # The following are commented out but kept for reference:
        # q_yt_log_p_yo = torch.zeros_like(etas)  # [N, H, W]
        # q_yt_log_q_yt = torch.zeros_like(etas)  # [N, H, W]

        for ell in range(C):
            q_yt_k = torch.where(targets[:, ell, ...] == 1, 1 - rs, rs / (self.class_num - 1))  # [N, H, W]
            # p_yo_k = torch.where(targets[:, ell, ...] == 1, 1 - rs, rs / (self.class_num - 1))  # [N, H, W]

            # fmt: off
            q_yt_k_log_p_yt_k = q_yt_k * log_p_yt[:, ell]  # [N, H, W]
            # q_yt_k_log_p_yo_k = q_yt_k * torch.log(p_yo_k)  # [N, H, W]
            # q_yt_k_log_q_yt_k = q_yt_k * torch.log(q_yt_k)  # [N, H, W]
            # fmt: on

            # Sum over the class dimension.
            q_yt_log_p_yt = q_yt_log_p_yt + q_yt_k_log_p_yt_k  # [N, H, W]
            # q_yt_log_p_yo = q_yt_log_p_yo + q_yt_k_log_p_yo_k  # [N, H, W]
            # q_yt_log_q_yt = q_yt_log_q_yt + q_yt_k_log_q_yt_k  # [N, H, W]

            q_yt[:, ell] = q_yt_k.detach()

        return q_yt_log_p_yt.mean(), q_yt #, q_yt_log_p_yo, q_yt_log_q_yt

    def _save_postdist(self, post_dists, m, G, epoch, end_estep, image_id):
        """
        Save a few posterior distributions for visualization purposes.

        Args:
            post_dists (Tensor): Posterior distributions to be saved.
            m (Tensor): Posterior mean parameter.
            G (Tensor): Posterior variance parameter.
            epoch (int): Current epoch number.
            end_estep (bool): Flag indicating the end of the E-step.
            image_id (list): List of image IDs.
        """
        if (epoch + 1) % 10 == 0 and end_estep:
            for i, imgid in enumerate(image_id):
                if imgid in self.cfg.data.save_imageids:
                    postdist_savepath = f"{self.cfg.utils.saveimageids_dir}/post_dist/{imgid}/epoch{epoch}.pth"
                    torch.save(post_dists[i], postdist_savepath)
                    params_savepath = f"{self.cfg.utils.saveimageids_dir}/params/{imgid}/epoch{epoch}.pth"
                    torch.save(torch.stack([m[i], G[i]]), params_savepath)
                    print(f"Saved posterior distribution for image {imgid} at epoch {epoch}")

    def _temperature_softmax(self, x, dim, T=1.0):
        """
        Apply a temperature-scaled softmax.

        Args:
            x (Tensor): Input tensor.
            dim (int): Dimension along which to apply softmax.
            T (float): Temperature parameter.
        Returns:
            Tensor: Output after applying temperature softmax.
        """
        return torch.exp(x / T) / torch.sum(torch.exp(x / T), dim=dim, keepdim=True)

    def forward(self, images, targets, params, image_id, epoch=None, end_estep=False):
        """
        Perform forward pass and compute predictions and loss.

        Args:
            images (Tensor): Input images.
            targets (Tensor): Ground-truth labels.
            params (Tensor): Posterior parameters with shape [B, 2, H, W] (2 for (m, Γ)).
            image_id (list): List of image IDs.
            epoch (int, optional): Current epoch number.
            end_estep (bool, optional): Flag indicating the end of the E-step.
        Returns:
            Tuple: (predictions, total_loss, loss_components)
                - predictions: Model probability outputs.
                - total_loss: Total loss value.
                - loss_components: Tuple of individual loss components (term1, term2, term3, term4, term5).
        """
        # Obtain model output and apply temperature-scaled softmax
        predictions = self.model(images).float()
        predictions = self._temperature_softmax(predictions, dim=1)

        # Extract posterior parameters (m, Γ)
        m, G = params[:, 0, :, :], params[:, 1, :, :]
        G = F.softplus(G)  # Ensure non-negative variance

        # Compute covariance-related terms for prior and posterior
        term2 = self._second_term(m, G)
        term5_prenorm = (self._log_detGamma(G + epsilon) +
                         (self.imsize * self.imsize) * (math.log(math.pi * 2) + 1)) / 2
        term5 = (term5_prenorm / (self.imsize * self.imsize)).mean()

        # Compute base terms using our internal _base_terms method.
        term1, q_yt = self._base_terms(predictions, targets, m, G, image_id, epoch)
        # print('term1, q_yt :', term1.shape, q_yt.shape)
        # The following commented-out derivation includes term3 and term4:
        # total_loss = - (term1 + term2 - term3 + term4 + term5)
        # For binary segmentation, term3 and term4 may cancel out or be set to zero.
        term3 = 0.
        term4 = 0.
        total_loss = - (term1 + term2 - term3 + term4 + term5)

        # Uncomment the following line if term3 and term4 are computed:
        # return predictions, total_loss, (term1.item(), term2.item(), term3.item(), term4.item(), term5.item())
        return predictions, total_loss, (term1.item(), term2.item(), 0., 0., term5.item())
