import torch
from robustness_eval.white_box_attack import *

class AcousticSystem_robust(torch.nn.Module):

    def __init__(self, 
                 classifier: torch.nn.Module, 
                 transform, 
                 defender_wav: torch.nn.Module=None,
                 defender_spec: torch.nn.Module=None,
                 defense_type: str='wave',
                 defense_method: str='DualPure'
                 ):
        super().__init__()

        '''
            the whole audio system: audio -> prediction probability distribution
            
            *defender: audio -> audio or spectrogram -> spectrogram
            *transform: audio -> spectrogram
            *classifier: spectrogram -> prediction probability distribution or 
                            audio -> prediction probability distribution
        '''

        self.classifier = classifier
        self.transform = transform
        self.defender_wav = defender_wav
        self.defender_spec = defender_spec
        self.defense_type = defense_type
        self.defense_method = defense_method
        self.defense_spec = None
        if self.defense_method == 'DiffSpec':
            self.defense_spec = 'spec'

    def forward(self, x, defend=True):

        # if 0.9 * x.max() > 1 and 0.9 * x.min() < -1:
        #     x = x / (2**15)

        # defense on waveform
        if defend == True and self.defender_wav is not None and self.defense_type == 'wave':
            if self.defense_method == 'DualPure':
                output = self.defender_wav(x)
                self.defense_spec = 'spec'
            elif self.defense_method == 'AudioPure' or self.defense_method == 'DDPM' \
                or self.defense_method == 'DiffNoise' or self.defense_method == 'DiffRev' or self.defense_method == 'OneShot':
                output = self.defender_wav(x)
        else: 
            output = x

        # convert waveform to spectrogram 
        if self.transform is not None: 
            output = self.transform(output)     

        # defense on spectrogram
        if defend == True and self.defender_spec is not None and self.defense_spec == 'spec':
            output = self.defender_spec(output)
        else: 
            output = output
        
        # give prediction of spectrogram
        output = self.classifier(output)

        return output
