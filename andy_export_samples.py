# andy_export_samples.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import utils

from transforms import *
from datasets.sc_dataset import *
from audio_models.create_model import *
from robustness_eval.white_box_attack import AudioAttack
from robustness_eval.black_box_attack import FAKEBOB
from torchvision.transforms import Compose

from diffusion_models.diffwave_sde import RevDiffWave
from diffusion_models.improved_diffusion_sde import RevImprovedDiffusion

from andy_diffdef_audio import MultiDiff

class WaveClassifier(torch.nn.Module):
    """Takes waveform [B,1,T], converts to mel, feeds base classifier (expects NCHW)."""
    def __init__(self, base: torch.nn.Module, to_mel_fn):
        super().__init__()
        self.base = base
        self.to_mel = to_mel_fn
    def forward(self, x_wave: torch.Tensor):
        spec = self.to_mel(x_wave)              # [B, M, T']
        if spec.dim() == 3:                     # add channel if missing
            spec = spec.unsqueeze(1)            # [B, 1, M, T']
        return self.base(spec)

def to_cuda(x):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_tensor_like(x, ref):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=ref.dtype, device=ref.device)
    return x

def clamp_wave(x):
    return torch.clamp(x, -1.0, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="root with {train,valid,test} OR a split dir")
    ap.add_argument("--classifier_path", required=True)
    ap.add_argument("--diffwav_path", required=True)
    ap.add_argument("--diffspec_path", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--num", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    #  MultiDiff
    ap.add_argument("--t", type=int, default=50)
    ap.add_argument("--sample_step", type=int, default=1)
    ap.add_argument("--rand_inits", type=int, default=8)
    ap.add_argument("--select", choices=["mse_wave","mse_mel","logit_margin"], default="logit_margin")
    ap.add_argument("--bpda_mode", choices=["identity","surrogate"], default="surrogate")
    ap.add_argument("--bpda_t", type=int, default=5)
    # ap.add_argument("--seed_eta", type=float, default=0.02) 
    # Attack 
    ap.add_argument("--pgd_eps", type=float, default=0.002)
    ap.add_argument("--pgd_iters", type=int, default=10)
    ap.add_argument("--fakebob_conf", type=float, default=0.5)
    ap.add_argument("--fakebob_iter", type=int, default=200)
    ap.add_argument("--fakebob_spd", type=int, default=20)
    ap.add_argument("--fakebob_spd_bs", type=int, default=10)
    args = ap.parse_args()

    use_gpu = torch.cuda.is_available()
    print("CUDA:", use_gpu)

    # Classifier
    print(f"Loading classifier from {args.classifier_path}")
    Classifier = create_model(args.classifier_path).eval()
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        Classifier.cuda()

    # Data
    transform = Compose([LoadAudio(), FixAudioLength()])
    ds = SC09Dataset(folder=args.data_path, transform=transform, num_per_class=args.num)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    pin_memory=use_gpu, num_workers=2)

    # Spectrogram transform (for classifier input + mse_mel if chosen)
    import torchaudio
    MelSpecTrans = torchaudio.transforms.MelSpectrogram(
        n_fft=2048, hop_length=512, n_mels=32, norm='slaney',
        pad_mode='constant', mel_scale='slaney'
    )
    Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
    if use_gpu:
        MelSpecTrans = MelSpecTrans.cuda()
        Amp2DB = Amp2DB.cuda()
    Wave2Spect = Compose([MelSpecTrans, Amp2DB])

    # Wrapper so logit_margin works on waveforms
    ClsForMargin = WaveClassifier(Classifier, Wave2Spect).eval()
    if use_gpu:
        ClsForMargin.cuda()

    # Baseline purifiers (wave + spec)
    class ArgsShim: pass
    shim = ArgsShim()
    shim.ddpm_config = 'configs/config.json'
    shim.diffwav_path = args.diffwav_path
    shim.diffspec_path = args.diffspec_path
    shim.sample_step = args.sample_step
    shim.t = args.t
    shim.t_delta = 0
    shim.rand_t = False
    shim.diffusion_type = 'ddpm'
    shim.score_type = 'guided_diffusion'
    shim.use_bm = True
    shim.defense = 'AudioPure'  # label only

    AP_wav = RevDiffWave(shim).eval()                 # AudioPure waveform purifier
    DP_wav = RevDiffWave(shim).eval()                 # DualPure waveform stage (same module type)
    SP_spec = RevImprovedDiffusion(shim).eval()       # spectrogram purifier (unused for audio export)

    if use_gpu:
        AP_wav.cuda(); DP_wav.cuda(); SP_spec.cuda()

    # MultiDiff (reuse the already-loaded DiffWave purifier to avoid duplicate load)
    DD_wav = MultiDiff(
        purifier=AP_wav,
        reverse_timestep=args.t,
        sample_step=args.sample_step,
        rand_inits=args.rand_inits,
        select=args.select,
        classifier=ClsForMargin if args.select == "logit_margin" else None,
        to_mel_fn=Wave2Spect if args.select == "mse_mel" else Wave2Spect,  # keep available
        bpda_mode=args.bpda_mode,
        bpda_t=args.bpda_t,
        #seed_eta=args.seed_eta
    ).eval()
    if use_gpu:
        DD_wav.cuda()

    # Attackers (attack the raw classifier pipeline: no defense)
    class AcousticSystemBare(torch.nn.Module):
        def __init__(self, classifier, wave2spec):
            super().__init__()
            self.classifier = classifier
            self.wave2spec = wave2spec
        def forward(self, x, defend=True):
            spec = self.wave2spec(x)        # [B, M, T]
            if spec.dim() == 3:
                spec = spec.unsqueeze(1)    # [B, 1, M, T]
            return self.classifier(spec)

    AS_none = AcousticSystemBare(Classifier, Wave2Spect).eval()
    if use_gpu:
        AS_none.cuda()

    PGD = AudioAttack(model=AS_none,
                      eps=args.pgd_eps, norm='linf',
                      max_iter_1=args.pgd_iters, max_iter_2=0,
                      learning_rate_1=args.pgd_eps/5,
                      eot_attack_size=1, eot_defense_size=1, verbose=0)

    FB = FAKEBOB(model=AS_none, task='SCR', targeted=False, verbose=0,
                 confidence=args.fakebob_conf, epsilon=args.pgd_eps,
                 max_lr=0.001, min_lr=1e-6, max_iter=args.fakebob_iter,
                 samples_per_draw=args.fakebob_spd,
                 samples_per_draw_batch_size=args.fakebob_spd_bs,
                 batch_size=1)

    ensure_dir(args.save_dir)

    exported = 0
    for batch in dl:
        if exported >= args.num:
            break

        wav = batch['samples']            # [B,T]
        wav = torch.unsqueeze(wav, 1)     # [B,1,T]
        y = batch['target']
        base = os.path.splitext(os.path.basename(batch['path'][0]))[0]
        outdir = os.path.join(args.save_dir, base)
        ensure_dir(outdir)

        wav = to_cuda(wav).float()
        y = to_cuda(y)

        # --- Clean ---
        utils.audio_save(wav[0].detach().cpu(), path=outdir, name="clean.wav")

        # --- PGD ---
        wav_pgd, _ = PGD.generate(x=wav, y=y, targeted=False)
        wav_pgd = to_tensor_like(wav_pgd, wav)
        wav_pgd = clamp_wave(wav_pgd)
        utils.audio_save(wav_pgd[0].detach().cpu(), path=outdir, name="pgd.wav")

        # --- FAKEBOB ---
        wav_fb, _ = FB.generate(x=wav, y=y, targeted=False)
        wav_fb = to_tensor_like(wav_fb, wav)
        wav_fb = clamp_wave(wav_fb)
        utils.audio_save(wav_fb[0].detach().cpu(), path=outdir, name="fakebob.wav")

        # --- Purify (clean) ---
        with torch.no_grad():
            ap_clean = clamp_wave(AP_wav(wav))
            dp_clean = clamp_wave(DP_wav(wav))
            dd_clean = clamp_wave(DD_wav(wav))
        utils.audio_save(ap_clean[0].cpu(), path=outdir, name="audiopure_clean.wav")
        utils.audio_save(dp_clean[0].cpu(), path=outdir, name="dualpure_wave_clean.wav")
        utils.audio_save(dd_clean[0].cpu(), path=outdir, name="multidiff_clean.wav")

        # --- Purify (PGD) ---
        with torch.no_grad():
            ap_pgd = clamp_wave(AP_wav(wav_pgd))
            dp_pgd = clamp_wave(DP_wav(wav_pgd))
            dd_pgd = clamp_wave(DD_wav(wav_pgd))
        utils.audio_save(ap_pgd[0].cpu(), path=outdir, name="audiopure_pgd.wav")
        utils.audio_save(dp_pgd[0].cpu(), path=outdir, name="dualpure_wave_pgd.wav")
        utils.audio_save(dd_pgd[0].cpu(), path=outdir, name="multidiff_pgd.wav")

        # --- Purify (FAKEBOB) ---
        with torch.no_grad():
            ap_fb  = clamp_wave(AP_wav(wav_fb))
            dp_fb  = clamp_wave(DP_wav(wav_fb))
            dd_fb  = clamp_wave(DD_wav(wav_fb))
        utils.audio_save(ap_fb[0].cpu(), path=outdir, name="audiopure_fakebob.wav")
        utils.audio_save(dp_fb[0].cpu(), path=outdir, name="dualpure_wave_fakebob.wav")
        utils.audio_save(dd_fb[0].cpu(), path=outdir, name="multidiff_fakebob.wav")

        print(f"[saved] {outdir}")
        exported += 1

if __name__ == "__main__":
    main()
