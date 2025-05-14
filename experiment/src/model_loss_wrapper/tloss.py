# Copyright 2023 University of Basel and Lucerne University of Applied Sciences and Arts Authors 
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


__author__ = "Alvaro Gonzalez-Jimenez"
__maintainer__ = "Alvaro Gonzalez-Jimenez"
__email__ = "alvaro.gonzalezjimenez@unibas.ch"
__license__ = "Apache License, Version 2.0"
__date__ = "2023-07-25"

import os
import numpy as np
import torch
import torch.nn as nn


class TLoss(nn.Module):
    def __init__(
        self,
        model,
        image_size: int = 512,
        nu: float = 1.0,
        epsilon: float = 1e-8,
        reduction: str = "mean",
    ):
        """
        Implementation of the TLoss.

        Args:
            config: Configuration object for the loss.
            nu (float): Value of nu.
            epsilon (float): Value of epsilon.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                             'none': no reduction will be applied,
                             'mean': the sum of the output will be divided by the number of elements in the output,
                             'sum': the output will be summed.
        """
        super().__init__()
        # self.config = config
        # gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model
        self.D = torch.tensor(
            (image_size * image_size),
            dtype=torch.float,
            # device=device,
        )
 
        self.lambdas = torch.ones(
            (image_size, image_size),
            dtype=torch.float,
            # device=device,
        )
        self.nu = nn.Parameter(
            torch.tensor(nu, dtype=torch.float)
        ) #.to(gpu_id)
        self.epsilon = torch.tensor(epsilon, dtype=torch.float)
        self.reduction = reduction

    def _temperature_softmax(self, x, dim, T=1.0):
        return torch.exp(x / T) / torch.sum(torch.exp(x / T), dim=dim, keepdim=True)

    def forward(
        self, images, targets, img_names, device
    ):
        """
        Args:
            images (torch.Tensor): Models prediction, size (B x C x W x H).
            targets (torch.Tensor): Ground truth, size (B x W x H).

        Returns:
            torch.Tensor: Total loss value.
        """
        predictions = self.model(images).float()
        predictions = self._temperature_softmax(predictions, 1)
        # predictions_squeezed = predictions[:,1,...] # Binary segmentation！！！
        # print("predictions.shape :", predictions.shape)
        # print("targets.shape :", targets.shape)

        delta_i = predictions - targets
        sum_nu_epsilon = torch.exp(self.nu) + self.epsilon
        first_term = -torch.lgamma((sum_nu_epsilon + self.D) / 2)
        second_term = torch.lgamma(sum_nu_epsilon / 2)
        third_term = -0.5 * torch.sum(self.lambdas + self.epsilon)
        fourth_term = (self.D / 2) * torch.log(torch.tensor(np.pi))
        fifth_term = (self.D / 2) * (self.nu + self.epsilon)

        delta_squared = torch.pow(delta_i, 2)
        lambdas_exp = torch.exp(self.lambdas + self.epsilon).to(device)
        numerator = delta_squared * lambdas_exp
        numerator = torch.sum(numerator, dim=(1, 2))

        fraction = numerator / sum_nu_epsilon
        sixth_term = ((sum_nu_epsilon + self.D) / 2) * torch.log(1 + fraction)

        total_losses = (
            first_term
            + second_term
            + third_term
            + fourth_term
            + fifth_term
            + sixth_term
        )

        if self.reduction == "mean":
            return predictions, total_losses.mean()
        elif self.reduction == "sum":
            return predictions, total_losses.sum()
        elif self.reduction == "none":
            return predictions, total_losses
        else:
            raise ValueError(
                f"The reduction method '{self.reduction}' is not implemented."
            )
