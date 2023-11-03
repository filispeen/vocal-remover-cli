import os

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from lib import dataset
from lib import nets
from lib import spec_utils

def start_inference(a_pretrained_model, a_gpu, a_input, a_sr, a_hop_length, a_postprocess, a_output_image, a_window_size):
    print('loading model...', end=' ')
    device = torch.device('cpu')
    model = nets.CascadedASPPNet()
    model.load_state_dict(torch.load(a_pretrained_model, map_location=device))
    if torch.cuda.is_available() and a_gpu >= 0:
        device = torch.device('cuda:{}'.format(a_gpu))
        model.to(device)
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        a_input, a_sr, False, dtype=np.float32, res_type='kaiser_fast')
    print('done')

    print('wave source stft...', end=' ')
    X = spec_utils.calc_spec(X, a_hop_length)
    X, phase = np.abs(X), np.exp(1.j * np.angle(X))
    coeff = X.max()
    X /= coeff
    print('done')

    offset = model.offset
    l, r, roi_size = dataset.make_padding(X.shape[2], a_window_size, offset)
    X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')

    masks = []
    model.eval()
    with torch.no_grad():
        for j in tqdm(range(int(np.ceil(X.shape[2] / roi_size)))):
            start = j * roi_size
            X_window = X_pad[None, :, :, start:start + a_window_size]
            pred = model.predict(torch.from_numpy(X_window).to(device))
            pred = pred.detach().cpu().numpy()
            masks.append(pred[0])

    mask = np.concatenate(masks, axis=2)[:, :, :X.shape[2]]
    if a_postprocess:
        vocal_pred = X * (1 - mask) * coeff
        mask = spec_utils.mask_uninformative(mask, vocal_pred)
    inst_pred = X * mask * coeff
    vocal_pred = X * (1 - mask) * coeff

    if a_output_image:
        norm_mask = np.uint8((1 - mask) * 255)
        canvas = np.zeros((norm_mask.shape[1], norm_mask.shape[2], 3))
        canvas[:, :, 1] = norm_mask[0]
        canvas[:, :, 2] = norm_mask[1]
        canvas[:, :, 0] = np.max(norm_mask, axis=0)
        cv2.imwrite('mask.png', canvas[::-1])

    basename = os.path.splitext(os.path.basename(a_input))[0]

    print('instrumental inverse stft...', end=' ')
    wav = spec_utils.spec_to_wav(inst_pred, phase, a_hop_length)
    print('done')
    sf.write('{}_Instrumental.wav'.format(basename), wav.T, sr)

    print('vocal inverse stft...', end=' ')
    wav = spec_utils.spec_to_wav(vocal_pred, phase, a_hop_length)
    print('done')
    sf.write('{}_Vocal.wav'.format(basename), wav.T, sr)