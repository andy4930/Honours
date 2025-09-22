import torch
from robustness_eval.white_box_attack import *

class AcousticSystem_robust(torch.nn.Module):

    def __init__(self, 
                 classifier: torch.nn.Module, 
                 transform, 
                 defender_wav: torch.nn.Module=None,
                 defender_spec: torch.nn.Module=None,
                 defense_type: str='wave',
                 defense_method: str='DualPure'):
        super().__init__()

        """
        audio pipeline:
          defender_wav:  audio -> audio
          transform:     audio -> spectrogram
          defender_spec: spectrogram -> spectrogram
          classifier:    spectrogram -> logits
        """
        self.classifier = classifier
        self.transform = transform
        self.defender_wav = defender_wav
        self.defender_spec = defender_spec
        self.defense_type = defense_type
        self.defense_method = defense_method

        # ORIGINAL BEHAVIOR: spec defense is OFF unless explicitly enabled
        self.defense_spec = None
        if self.defense_method == 'DiffSpec':
            self.defense_spec = 'spec'

    def forward(self, x, defend=True):
        # ---- Waveform defense (unchanged logic) ----
        if defend and self.defender_wav is not None and self.defense_type == 'wave':
            if self.defense_method == 'DualPure':
                # DualPure: do wave defense AND enable the next spec stage
                output = self.defender_wav(x)
                self.defense_spec = 'spec'
            elif self.defense_method in ['AudioPure', 'DDPM', 'DiffNoise', 'DiffRev', 'OneShot']:
                output = self.defender_wav(x)
            elif self.defense_method == 'MultiDiff':
                output = self.defender_wav(x)
            else:
                output = x
        else:
            output = x

        # ---- Waveform -> Spectrogram (unchanged) ----
        if self.transform is not None:
            output = self.transform(output)

        # ---- Spectrogram defense (unchanged condition) ----
        if defend and self.defender_spec is not None and self.defense_spec == 'spec':
            output = self.defender_spec(output)

        # ---- Classify ----
        output = self.classifier(output)
        return output
