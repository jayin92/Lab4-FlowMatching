from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .scheduler import extract

class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler
        self.predictor = kwargs.get("predictor", "noise")


    def get_loss_noise(self, x0, class_label=None, noise=None):
        B = x0.shape[0]
        t = self.var_scheduler.uniform_sample_t(B, x0.device)  # (B,)
        x_t, eps = self.var_scheduler.add_noise(x0, t, eps=noise)
        eps_pred = self.network(x_t, t, class_label) if class_label is not None else self.network(x_t, t)
        return F.mse_loss(eps_pred, eps)
    
    def get_loss_x0(self, x0, class_label=None, noise=None):
        # Here we implement the "predict x0" version.
        # 1. Sample a timestep and add noise to get (x_t, noise).
        # 2. Pass (x_t, timestep) into self.network, where the output should represent the clean sample x0_pred.
        # 3. Compute the loss as MSE(predicted x0_pred, ground-truth x0).
        B = x0.shape[0]
        t = self.var_scheduler.uniform_sample_t(B, x0.device)  # (B,)
        x_t, eps = self.var_scheduler.add_noise(x0, t, eps=noise)
        
        x0_pred = self.network(x_t, t, class_label) if class_label is not None else self.network(x_t, t)
        
        loss = F.mse_loss(x0_pred, x0)
        return loss
    
    def get_loss_mean(self, x0, class_label=None, noise=None):
        # Here we implement the "predict mean" version.
        # 1. Sample a timestep and add noise to get (x_t, noise).
        # 2. Pass (x_t, timestep) into self.network, where the output should represent the posterior mean μθ(x_t, t).
        # 3. Compute the *true* posterior mean from the closed-form DDPM formula (using x0, x_t, noise, and scheduler terms).
        # 4. Compute the loss as MSE(predicted mean, true mean).
        #
        # Note: This assumes your `var_scheduler` has `posterior_mean_coef1` and `posterior_mean_coef2`
        # precomputed, which is standard for DDPM.
        B = x0.shape[0]
        t = self.var_scheduler.uniform_sample_t(B, x0.device)  # (B,)
        x_t, eps = self.var_scheduler.add_noise(x0, t, eps=noise)
        
        mean_pred = self.network(x_t, t, class_label) if class_label is not None else self.network(x_t, t)

        # Compute the true posterior mean q(x_{t-1} | x_t, x_0)
        # This is: \tilde{\mu}_t(x_t, x_0) = coef1 * x_0 + coef2 * x_t
        # where coef1 = (sqrt(\bar{\alpha}_{t-1}) * \beta_t) / (1 - \bar{\alpha}_t)
        # and     coef2 = (sqrt(\alpha_t) * (1 - \bar{\alpha}_{t-1})) / (1 - \bar{\alpha}_t)
        
        # We use the extract helper function to get the coefficients for the current timesteps t
        true_mean_coef1 = extract(self.var_scheduler.posterior_mean_coef1, t, x0.shape)
        true_mean_coef2 = extract(self.var_scheduler.posterior_mean_coef2, t, x_t.shape)
        
        true_mean = true_mean_coef1 * x0 + true_mean_coef2 * x_t
        
        loss = F.mse_loss(mean_pred, true_mean)
        return loss
    
    def get_loss(self, x0, class_label=None, noise=None):
        if self.predictor == "noise":
            return self.get_loss_noise(x0, class_label, noise)
        elif self.predictor == "x0":
            return self.get_loss_x0(x0, class_label, noise)
        elif self.predictor == "mean":
            return self.get_loss_mean(x0, class_label, noise)
        else:
            raise ValueError(f"Unknown predictor: {self.predictor}")

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 1.0,
    ):
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            assert class_label is not None, "class_label must be provided for classifier-free guidance"
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"

            # Specifically, given a tensor of shape (batch_size,) containing class labels,
            # create a tensor of shape (2*batch_size,) where the first half is filled with zeros (i.e., null condition).
            
            # Create the null condition (unconditional) tensor, usually zeros.
            null_cond = torch.zeros_like(class_label)
            
            # Concatenate the unconditional and conditional labels.
            # The network will process this [2*B, ...] tensor in one batch.
            class_label = torch.cat([null_cond, class_label])

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            
            # Ensure t is a tensor of shape (batch_size,)
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

            if do_classifier_free_guidance:
                # 1. Duplicate x_t and t to create a batch of size 2*B
                x_t_in = torch.cat([x_t] * 2) # [2*B, C, H, W]
                t_in = torch.cat([t_tensor] * 2) # [2*B]
                
                # 2. Call the network once with the combined inputs
                # class_label is already [2*B, ...] from the setup above
                net_out_combined = self.network(x_t_in, timestep=t_in, class_label=class_label)
                
                # 3. Split the output into unconditional and conditional parts
                # .chunk(2) splits the tensor into 2 chunks along dimension 0
                net_out_uncond, net_out_cond = net_out_combined.chunk(2)
                
                # 4. Apply the CFG formula:
                # \hat{\epsilon} = \epsilon_{uncond} + s \cdot (\epsilon_{cond} - \epsilon_{uncond})
                # This logic applies regardless of whether self.predictor is "noise", "x0", or "mean"
                net_out = net_out_uncond + guidance_scale * (net_out_cond - net_out_uncond)
                
            else:
                # Standard conditional or unconditional sampling
                if class_label is not None:
                    net_out = self.network(x_t, timestep=t_tensor, class_label=class_label)
                else:
                    net_out = self.network(x_t, timestep=t_tensor)

            # The scheduler step calculates x_{t-1} from x_t and the network output
            x_t_prev = self.var_scheduler.step(x_t, t, net_out, predictor=self.predictor)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]


    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
            } 
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
