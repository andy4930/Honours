import os
import argparse
import random
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import *
import torchaudio

from robustness_eval.black_box_attack import *
import utils

from andy_diffdef_audio import MultiDiff

# torch.manual_seed(42)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    '''SC09 classifier arguments'''
    parser.add_argument("--data_path", help='sc09 dataset folder')
    parser.add_argument("--classifier_path", help='dir of saved classifier model')
    parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='input of NN')
    parser.add_argument("--num_per_class", type=int, default=10)

    '''DiffWave-VPSDE arguments'''
    parser.add_argument('--ddpm_config', type=str, default='configs/config.json', help='JSON file for configuration')
    parser.add_argument('--diffwav_path', type=str, default=None, help='dir of diffusion model checkpoint')
    parser.add_argument('--diffspec_path', type=str, default=None, help='dir of diffusion model checkpoint')
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=3, help='diffusion steps, control the sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=0, help='perturbation range of sampling noise scale; set to 0 by default')
    parser.add_argument('--rand_t', action='store_true', default=False, help='decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='sde', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde, ddpm]')
    parser.add_argument('--use_bm', action='store_true', default=True, help='whether to use brownian motion')
    
    '''attack arguments'''
    parser.add_argument('--attack', type=str, choices=['PGD', 'FAKEBOB'], default='PGD')
    parser.add_argument('--defense', type=str, choices=['DualPure', 'AudioPure', 'DDPM', 'DiffNoise', 'DiffRev', 'OneShot', 'DiffSpec', 'MultiDiff','None'], default='None')
    parser.add_argument('--defense_type', type=str, choices=['wave','spec'], default='wave',
                    help='where to apply the defense (waveform or spectrogram)') # Added by Andy
    parser.add_argument('--bound_norm', type=str, choices=['linf', 'l2'], default='linf')
    parser.add_argument('--eps', type=float, default=0.002)  # l2=0.253
    parser.add_argument('--max_iter_1', type=int, default=10)
    parser.add_argument('--max_iter_2', type=int, default=0)
    parser.add_argument('--eot_attack_size', type=int, default=1, help='EOT size of attack')
    parser.add_argument('--eot_defense_size', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--rand_inits', type=int, default=3) # Added by Andy
    parser.add_argument('--select', type=str, choices=['mse_wave','mse_mel','logit_margin'],
                    default='mse_wave', help='candidate scoring for MultiDiff') #Added by Andy
    parser.add_argument('--bpda_mode', type=str, choices=['identity','surrogate'], default='surrogate') # Added by Andy
    parser.add_argument('--bpda_t', type=int, default=5) # Added by Andy

    '''device arguments'''
    parser.add_argument("--dataload_workers_nums", type=int, default=4, help='number of workers for dataloader')
    parser.add_argument("--batch_size", type=int, default=1, help='batch size')
    parser.add_argument('--gpu', type=int, default=0)

    '''file saving arguments'''
    parser.add_argument('--save_path', default=None)

    args = parser.parse_args()

    '''device setting'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    print('gpu id: {}'.format(args.gpu))

    '''SC09 classifier setting'''
    from transforms import *
    from datasets.sc_dataset import *
    from audio_models.create_model import *

    # Added by Andy - for -- select logit_margin
    class WaveClassifier(torch.nn.Module):
        # Wraps a spectrogram classifier so it can accept waveform [B,1,T].
        def __init__(self, base: torch.nn.Module, to_mel_fn):
            super().__init__()
            self.base = base
            self.to_mel = to_mel_fn
        def forward(self, x_wave: torch.Tensor):
            spec = self.to_mel(x_wave)          # [B, M, T’]
            if spec.dim() == 3:
                spec = spec.unsqueeze(1)        # [B, 1, M, T’]
            return self.base(spec)
    
    print('create classifier from {}'.format(args.classifier_path))
    Classifier = create_model(args.classifier_path)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        Classifier.cuda()

    transform = Compose([LoadAudio(), FixAudioLength()])
    test_dataset = SC09Dataset(folder=args.data_path, transform=transform, num_per_class=args.num_per_class)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=None, shuffle=False, 
                                pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    criterion = torch.nn.CrossEntropyLoss()

    '''preprocessing setting (if use acoustic features like mel-spectrogram)'''
    n_mels = 32
    if args.classifier_input == 'mel40':
        n_mels = 40
    MelSpecTrans = torchaudio.transforms.MelSpectrogram(n_fft=2048, hop_length=512, n_mels=n_mels, norm='slaney', pad_mode='constant', mel_scale='slaney')
    Amp2DB = torchaudio.transforms.AmplitudeToDB(stype='power')
    Wave2Spect = Compose([MelSpecTrans.cuda(), Amp2DB.cuda()])

    # Added by Andy
    ClsForMargin = WaveClassifier(Classifier, Wave2Spect).eval()
    if use_gpu:
        ClsForMargin.cuda()

    defense_method_all = ['DualPure', 'AudioPure', 'DDPM', 'DiffNoise', 'DiffRev', 'OneShot', 'DiffSpec', 'MultiDiff']
    '''defense setting'''
    from acoustic_system import AcousticSystem_robust
    if args.defense == 'None':
        if Classifier._get_name() == 'M5': # M5Net takes the raw audio as input
            AS_MODEL = AcousticSystem_robust(classifier=Classifier, transform=None, defender=None)
        else: 
            AS_MODEL = AcousticSystem_robust(classifier=Classifier, transform=Wave2Spect, defender_wav=None, defender_spec=None, defense_method='None') # Changed by Andy
        print('classifier model: {}'.format(Classifier._get_name()))
        print('defense: None')

    else:
        if args.defense in defense_method_all:
            from diffusion_models.diffwave_sde import *
            from diffusion_models.improved_diffusion_sde import *
            Defender_wav_base = RevDiffWave(args)
            Defender_spec_base = RevImprovedDiffusion(args)
        else:
            raise NotImplementedError(f'Unknown defense: {args.defense}!')
        
        defender_wav = None
        defender_spec = None

        if args.defense == 'AudioPure':
            defender_wav = Defender_wav_base

        elif args.defense == 'DualPure':
            defender_wav = Defender_wav_base
            defender_spec = Defender_spec_base

        elif args.defense == 'DiffSpec':
            defender_spec = Defender_spec_base

        elif args.defense == 'DDPM' or args.defense == 'DiffNoise' or args.defense == 'DiffRev' or args.defense == 'OneShot':
            defender_wav = Defender_wav_base

        elif args.defense == 'MultiDiff':
            # Waveform variant
            if args.defense_type == 'wave':
                defender_wav = MultiDiff(
                    purifier=Defender_wav_base,
                    reverse_timestep=args.t,
                    sample_step=args.sample_step,
                    rand_inits=args.rand_inits,
                    select=args.select,
                    classifier=ClsForMargin if args.select == 'logit_margin' else None,
                    to_mel_fn=Wave2Spect if args.select == 'mse_mel' else None,
                    bpda_mode=args.bpda_mode,
                    bpda_t=args.bpda_t,
                )
            # Spectrogram variant (optional)
            # elif args.defense_type == 'spec':
            #     defender_spec = MultiDiff(
            #         purifier=Defender_spec_base,
            #         reverse_timestep=args.t,
            #         sample_step=args.sample_step,
            #         rand_inits=args.rand_inits,
            #         select='mse_wave',
            #         classifier=None,
            #         to_mel_fn=None,
            #         bpda_mode=args.bpda_mode,
            #         bpda_t=args.bpda_t,
            #    )
            else:
                raise ValueError(f"Unknown defense_type: {args.defense_type}")

        
        if Classifier._get_name() == 'M5':  # raw-audio classifier
            AS_MODEL = AcousticSystem_robust(
                classifier=Classifier, transform=None,
                defender_wav=defender_wav, defender_spec=defender_spec,
                defense_type=args.defense_type, defense_method=args.defense
            )
        else:
            AS_MODEL = AcousticSystem_robust(
                classifier=Classifier, transform=Wave2Spect,
                defender_wav=defender_wav, defender_spec=defender_spec,
                defense_type=args.defense_type, defense_method=args.defense
            )

        print('classifier model: {}'.format(Classifier._get_name()))

        if args.defense == 'Diffusion':
            print('defense: {} with t={}, s={}'.format(Defender_wav_base._get_name(), args.t, args.sample_step))
            print('diffusion type: {}'.format(args.diffusion_type))
        else:
            print('defense: {}'.format(Defender_wav_base._get_name()))
        
        if defender_wav is not None:
            print(f'defense (wave): {defender_wav._get_name() if hasattr(defender_wav, "_get_name") else type(defender_wav).__name__}')
        if defender_spec is not None:
            print(f'defense (spec): {defender_spec._get_name() if hasattr(defender_spec, "_get_name") else type(defender_spec).__name__}')
        if args.defense == 'MultiDiff':
            print(f'MultiDiff params: t={args.t}, step={args.sample_step}, rand_inits={args.rand_inits}, select={args.select}')

    AS_MODEL.eval()

    '''attack setting'''
    from robustness_eval.white_box_attack import *
    if args.attack == 'PGD':
        Attacker = AudioAttack(model=AS_MODEL, 
                                eps=args.eps, norm=args.bound_norm,
                                max_iter_1=args.max_iter_1, max_iter_2=0,
                                learning_rate_1=args.eps/5 if args.bound_norm=='linf' else args.eps/2, 
                                eot_attack_size=args.eot_attack_size,
                                eot_defense_size=args.eot_defense_size,
                                verbose=args.verbose)
        print('attack: {} with {}_eps={} & iter={} & eot={}-{}\n'\
            .format(args.attack, args.bound_norm, args.eps, args.max_iter_1, args.eot_attack_size, args.eot_defense_size))
    elif args.attack == 'FAKEBOB':
        eps = args.eps
        confidence = 0.5
        max_iter = 200
        samples_per_draw = 200 # or 20 
        samples_per_draw_batch_size = 10
        Attacker = FAKEBOB(model=AS_MODEL, task='SCR', targeted=False, verbose=args.verbose,
                           confidence=confidence, epsilon=eps, max_lr=0.001, min_lr=1e-6,
                           max_iter=max_iter, samples_per_draw=samples_per_draw, samples_per_draw_batch_size=samples_per_draw_batch_size, batch_size=args.batch_size)
        print('attack: {} with eps={} & confidence={} & iter={} & samples_per_draw={}\n'\
            .format(args.attack, eps, confidence, max_iter, samples_per_draw))
    else:
        raise AttributeError("this version does not support '{}' at present".format(args.attack))

    
    '''robustness eval'''
    from tqdm import tqdm
    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)

    correct_orig = 0
    correct_orig_denoised = 0
    correct_adv_1 = 0
    total = 0

    for batch in pbar:
        waveforms = batch['samples']
        waveforms = torch.unsqueeze(waveforms, 1)
        targets = batch['target']

        waveforms = waveforms.cuda()
        targets = targets.cuda()

        with torch.no_grad():
            '''original audio'''
            pred_clean = AS_MODEL(waveforms, False).max(1, keepdim=True)[1].squeeze()
        
            '''denoised original audio'''
            if AS_MODEL.defense_type == 'wave':
                if args.defense == 'None':
                    waveforms_defended = waveforms
                else:
                    waveforms_defended = AS_MODEL.defender_wav(waveforms)
                pred_defended = AS_MODEL(waveforms_defended, False).max(1, keepdim=True)[1].squeeze()
            elif AS_MODEL.defense_type == 'spec': 
                spectrogram = AS_MODEL.transform(waveforms)
                if args.defense == 'None':
                    spectrogram_defended = spectrogram
                else:
                    spectrogram_defended = AS_MODEL.defender_spec(spectrogram)
                pred_defended = AS_MODEL.classifier(spectrogram_defended).max(1, keepdim=True)[1].squeeze()

        '''adversarial audio'''
        waveforms_adv, attack_success = Attacker.generate(x=waveforms, y=targets, targeted=False)
        
        if isinstance(waveforms_adv, np.ndarray):
            if waveforms_adv.dtype == np.int16 and waveforms_adv.max() > 1 and waveforms_adv.min() < -1:
                    waveforms_adv = waveforms_adv / (2**15)
            waveforms_adv = torch.tensor(waveforms_adv, dtype=waveforms.dtype).to(waveforms.device)
        
        '''waveform/spectrogram saving'''
        if args.save_path is not None:
 
            clean_path = os.path.join(args.save_path,'clean', batch['path'][0].split('/')[-2])
            adv_path = os.path.join(args.save_path,'adv', batch['path'][0].split('/')[-2])

            if not os.path.exists(clean_path):
                os.makedirs(clean_path)
            if not os.path.exists(adv_path):
                os.makedirs(adv_path)

            for i in range(waveforms.shape[0]):
                
                audio_id = str(total + i).zfill(3)

                if AS_MODEL.defense_type == 'wave': 
                    '''original'''
                    # utils.audio_save(waveforms[i], path=clean_path, 
                    #                  name='{}_{}_clean.wav'.format(audio_id,targets[i].item()))
                    # # utils.audio_save(waveforms_defended[i], path=clean_path, 
                    # #                  name='{}_{}_clean_purified.wav'.format(audio_id,targets[i].item()))
                    # utils.audio_save(waveforms_adv[i], path=adv_path, 
                    #                  name='{}_{}_adv.wav'.format(audio_id,targets[i].item()))
                    # # utils.audio_save(waveforms_adv[i], path=adv_path, 
                    # #                  name='{}_{}_adv_purified.wav'.format(audio_id,targets[i].item()))
                    '''for adv-training'''
                    # print(batch['path'])
                    utils.audio_save(waveforms[i], path=clean_path, 
                                     name='{}_clean.wav'.format(batch['path'][0].split('/')[-1][:-4]))
                    # utils.audio_save(waveforms_defended[i], path=clean_path, 
                    #                  name='{}_{}_clean_purified.wav'.format(audio_id,targets[i].item()))
                    utils.audio_save(waveforms_adv[i], path=adv_path, 
                                     name='{}_adv.wav'.format(batch['path'][0].split('/')[-1][:-4]))
                    # utils.audio_save(waveforms_adv[i], path=adv_path, 
                    #                  name='{}_{}_adv_purified.wav'.format(audio_id,targets[i].item()))
                elif AS_MODEL.defense_type == 'spec':
                    utils.spec_save(spectrogram[i], path=clean_path, 
                                    name='{}_{}_clean.png'.format(audio_id,targets[i].item()))
                    utils.spec_save(spectrogram_defended[i], path=clean_path, 
                                    name='{}_{}_clean_purified.png'.format(audio_id,targets[i].item()))
                    # utils.spec_save(spectrogram_adv[i], path=adv_path, 
                    #                 name='{}_{}_adv.png'.format(audio_id,targets[i].item()))
                    # utils.spec_save(spectrogram_adv_defended[i], path=adv_path, 
                    #                 name='{}_{}_adv_purified.png'.format(audio_id,targets[i].item()))

        '''metrics output'''
        total += waveforms.shape[0]
        correct_orig += (pred_clean==targets).sum().item()
        correct_orig_denoised += (pred_defended==targets).sum().item()
        acc_orig = correct_orig / total * 100
        acc_orig_denoised = correct_orig_denoised / total * 100

        if isinstance(attack_success, tuple):
            correct_adv_1 += waveforms.shape[0] - torch.tensor(attack_success[0]).sum().item()
            acc_adv_1 = correct_adv_1 / total * 100
            pbar_info = {
                        'orig clean acc: ': '{:.4f}%'.format(acc_orig),
                        'denoised clean acc: ': '{:.4f}%'.format(acc_orig_denoised),
                        '{} robust acc: '.format(args.attack): '{:.4f}%'.format(acc_adv_1)
                        }
        else:
            correct_adv_1 += waveforms.shape[0] - torch.tensor(attack_success).sum().item()
            acc_adv_1 = correct_adv_1 / total * 100

            pbar_info = {
                        'orig clean acc: ': '{:.4f}%'.format(acc_orig),
                        'denoised clean acc: ': '{:.4f}%'.format(acc_orig_denoised),
                        '{} robust acc: '.format(args.attack): '{:.4f}%'.format(acc_adv_1)
                        }

        pbar.set_postfix(pbar_info)
        pbar.update(1)


    '''summary'''
    print('on {} test examples: '.format(total))
    print('original clean test accuracy: {:.4f}%'.format(acc_orig))
    print('denoised clean test accuracy: {:.4f}%'.format(acc_orig_denoised))
    print('{} robust test accuracy: {:.4f}%'.format(args.attack, acc_adv_1))

