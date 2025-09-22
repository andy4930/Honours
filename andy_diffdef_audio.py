# MultiDiff.py
import torch, torch.nn as nn
import torch.nn.functional as F

"""
    MultiDiff
    - K different versions of the adversarial inputs at an intermediate timestep
    - short reverse diffusion (few steps)
    - pick best candidate
"""
class MultiDiff(nn.Module):
    def __init__(self, purifier: nn.Module,
                 reverse_timestep: int = 50, # Starting time step for reverse process
                 sample_step: int = 1, # size of step between reverse timesteps
                 rand_inits: int = 8, # number of times to run the purification process with different random intial noise sample candidates
                 select: str = "logit_margin",   # Criterion for selecting the best candidate compares with adversarial input:  "mse_mel" | "mse_wave" | "logit_margin"
                 bpda_mode="surrogate", # Backward Pass Differentiable Approximation: "surrogate (tiny, differentiable reverse diffusion pass)" | "identity (gradient approx identity function)"
                 bpda_t=5, # Number of steps for the surrogate BPDA reverse pass
                 classifier: nn.Module = None, # Optional Classifier used for when select: = logit_margin
                 to_mel_fn=None): # Optional function to convert waveforms to mel spectrograms for when select = mse_mel
        super().__init__()
        self.purifier = purifier
        self.reverse_timestep = reverse_timestep
        self.sample_step = sample_step
        self.rand_inits = rand_inits
        self.select = select
        self.classifier = classifier
        self.to_mel_fn = to_mel_fn
        self.bpda_mode = bpda_mode
        self.bpda_t = bpda_t
        

    
    def forward(self, x_adv: torch.Tensor):
        B = x_adv.shape[0]
        best = None
        best_score = None

        # Takes a tensor and returns a single scalar value for each sample in the batch - used to calc scores
        def _per_sample_scalar(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(t.shape[0], -1).mean(dim=1)  # -> [B]

        for _ in range(self.rand_inits):
            # Generate random noise tensor
            z = torch.randn_like(x_adv)
            
            with torch.no_grad():
                # Perform reverse diffusion from random noise
                x_rec = self._reverse_short(
                    z, seed=x_adv, start_t=self.reverse_timestep, step=self.sample_step
                )

            # Select the best candidate based on the specified criterion 
            if self.select == "mse_mel" and self.to_mel_fn is not None:
                with torch.no_grad():
                    mel_rec = self.to_mel_fn(x_rec)
                    mel_adv = self.to_mel_fn(x_adv)
                    score = -_per_sample_scalar((mel_rec - mel_adv) ** 2)
            elif self.select == "logit_margin" and self.classifier is not None:
                with torch.no_grad():
                    logits = self.classifier(x_rec)
                    top2 = torch.topk(logits, 2, dim=1).values
                    score = (top2[:, 0] - top2[:, 1])
            else: #mse_wave
                with torch.no_grad():
                    score = -_per_sample_scalar((x_rec - x_adv) ** 2)

            if best is None:
                best, best_score = x_rec, score
            else:
                mask = (score > best_score).view(-1)  # [B]
                if mask.any():
                    best[mask] = x_rec[mask]
                    best_score[mask] = score[mask]

        # If bpda_mode is "identity" - ensure no graident flows through the best tensor and that gradient from output is passed directly to the input
        if self.bpda_mode == "identity":
            # forward = best, backward = identity wrt x_adv
            return best.detach() + x_adv - x_adv.detach()

        # If surrogate BPDA: forward uses best; backward uses grad of a tiny reverse pass
        if hasattr(self.purifier, "audio_editing_sample"):
            old_t = getattr(getattr(self.purifier, "args", object()), "t", None)
            if hasattr(self.purifier, "args") and hasattr(self.purifier.args, "t"):
                self.purifier.args.t = int(self.bpda_t)
            y_surr = self.purifier.audio_editing_sample(x_adv)  
            if hasattr(self.purifier, "args") and hasattr(self.purifier.args, "t") and old_t is not None:
                self.purifier.args.t = old_t
        elif hasattr(self.purifier, "image_editing_sample"):
            old_t = getattr(getattr(self.purifier, "args", object()), "t", None)
            if hasattr(self.purifier, "args") and hasattr(self.purifier.args, "t"):
                self.purifier.args.t = int(self.bpda_t)
            y_surr = self.purifier.image_editing_sample(x_adv)
            if hasattr(self.purifier, "args") and hasattr(self.purifier.args, "t") and old_t is not None:
                self.purifier.args.t = old_t
        else:
            # fallback
            return best.detach() + x_adv - x_adv.detach()

        # forward value = best; backward gradient = grad(y_surr wrt x_adv)
        return best.detach() + (y_surr - y_surr.detach())

    """
        Short reverse diffusion. 
        - For RevDiffWave we encode the input to x_t at timestep `start_t` and reverse from there. 
        - We use multiple restarts by calling it repeatedly.
    """
    def _reverse_short(self, z: torch.Tensor, seed: torch.Tensor, start_t: int, step: int):
        
        p = self.purifier
        p.eval()

        # Ensure tensors are on the purifier's device
        try:
            device = next(p.parameters()).device
        except StopIteration:
            device = seed.device
        z = z.to(device)
        seed = seed.to(device)

        # Cache original sampler args and override with new parameters for a short run
        old_t = getattr(getattr(p, "args", object()), "t", None)
        old_step = getattr(getattr(p, "args", object()), "sample_step", None)
        if hasattr(p, "args"):
            if hasattr(p.args, "t"):
                p.args.t = int(start_t)
            if hasattr(p.args, "sample_step"):
                p.args.sample_step = int(step)

        with torch.no_grad():
            if hasattr(p, "audio_editing_sample"):
                # add small Gaussian nudge
                eta = 0.02  # or 0.01
                seed = seed + eta * torch.randn_like(seed)
                seed = torch.clamp(seed, -1.0, 1.0)  # pipeline expects [-1,1]
                x0 = p.audio_editing_sample(seed)
            #elif hasattr(p, "image_editing_sample"):
                # Spec path (Improved Diffusion wrapper): expects spec/mel to edit
                #x0 = p.image_editing_sample(seed)
            else:
                # Last resort: try pure reverse from random noise ~ 10% PGD Robustness Accuracy 
                x0 = p(z)

        # Restore original args
        if hasattr(p, "args"):
            if old_t is not None and hasattr(p.args, "t"):
                p.args.t = old_t
            if old_step is not None and hasattr(p.args, "sample_step"):
                p.args.sample_step = old_step

        return x0
