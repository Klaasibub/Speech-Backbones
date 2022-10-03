import os
import warnings
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm

import torch
from torchaudio.transforms import Resample
import numpy as np

from scipy.io.wavfile import read, write
import pyloudnorm

from speechbrain.pretrained import EncoderClassifier

import params_ru as params
from stressrnn import StressRNN


METADATA_PATHS = [
    'filelists/ru_v3/train.txt',
    'filelists/ru_v3/val.txt',
    'filelists/ru_v3/test.txt'
]
MIN_LEN = 0.5
MAX_LEN = 16
SAMPLE_RATE = 16000

stress_rnn = StressRNN()

# torch 1.9 has a bug in the hub loading, this is a workaround
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
# careful: assumes 16kHz or 8kHz audio
silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False,
                                     onnx=False, verbose=False)

get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = utils


def normalize_audio(audio, sr, cut_silence=True):
    """
    one function to apply them all in an
    order that makes sense.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and getattr(params, 'cuda', True) else "cpu")

    # Normalize loudness
    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    loud_normed = pyloudnorm.normalize.loudness(audio, loudness, -30.0)
    peak = np.amax(np.abs(loud_normed))
    audio = np.divide(loud_normed, peak)
    audio = torch.Tensor(audio).to(device)

    if sr != SAMPLE_RATE:
        resample = Resample(orig_freq=sr, new_freq=SAMPLE_RATE).to(device)
        audio = resample(audio)

    if cut_silence:
        """
        https://github.com/snakers4/silero-vad
        """
        with torch.inference_mode():
            speech_timestamps = get_speech_timestamps(audio, silero_model, sampling_rate=SAMPLE_RATE)
        try:
            result = audio[speech_timestamps[0]['start']:speech_timestamps[-1]['end']]
            audio = result
        except IndexError:
            print("Audio might be too short to cut silences from front and back.")

    return audio.to("cpu")


def audio_to_wave_tensor(audio, sr, normalize=True):
    if normalize:
        return normalize_audio(audio, sr)
    else:
        if isinstance(audio, torch.Tensor):
            return audio
        else:
            return torch.Tensor(audio)


def process_audio(wavpath):
    return None
    sr, wave = read(wavpath)
    wave = torch.FloatTensor(wave.astype(np.float32)).numpy()
    dur_in_seconds = len(wave) / sr

    if not (MIN_LEN <= dur_in_seconds <= MAX_LEN):
        print(f"Excluding {wavpath} because of its duration of {round(dur_in_seconds, 2)} seconds.")
        return None

    try:
        # otherwise we get tons of warnings about an RNN not being in contiguous chunks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            norm_wave = audio_to_wave_tensor(normalize=True, audio=wave, sr=sr)
    except Exception as ex:
        print(f'Excluding {wavpath} when audio to tensor')
        print(ex)
        return None

    if not (MIN_LEN <= dur_in_seconds <= MAX_LEN):
        print(f"Excluding {wavpath} because of its duration of {round(dur_in_seconds, 2)} seconds.")
        return None

    norm_wave = torch.tensor(np.trim_zeros(norm_wave.numpy()))
    return norm_wave


def generate_embeds(metadata_path):
    device = torch.device("cuda" if torch.cuda.is_available() and getattr(params, 'cuda', True) else "cpu")
    with open(metadata_path, 'r', encoding='utf-8') as md:
        lines = md.readlines()
    wavpaths = {line.strip().split('|')[0]: line for line in lines}

    norm_waves = [(process_audio(wavpath), wavpath) for wavpath in tqdm(wavpaths.keys(), "Normalize audios")]
    # norm_waves = list(filter(lambda x: x[0] is not None, norm_waves))

    # # https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    # speaker_embedding_func_ecapa = EncoderClassifier.from_hparams(
    #     source="speechbrain/spkrec-ecapa-voxceleb",
    #     run_opts={"device": str(device)},
    #     savedir="Models/SpeakerEmbedding/speechbrain_speaker_embedding_ecapa"
    # )
    # # https://huggingface.co/speechbrain/spkrec-xvect-voxceleb
    # speaker_embedding_func_xvector = EncoderClassifier.from_hparams(
    #     source="speechbrain/spkrec-xvect-voxceleb",
    #     run_opts={"device": str(device)},
    #     savedir="Models/SpeakerEmbedding/speechbrain_speaker_embedding_xvector"
    # )
    with torch.no_grad():
        md_out = metadata_path.split('.', 1)[0] + "_embed_stress." + metadata_path.split('.', 1)[-1]
        with open(md_out, 'w', encoding='utf-8') as md:
            for wave, wavepath in tqdm(norm_waves, "Generate embeds"):
                # spk_emb_ecapa = speaker_embedding_func_ecapa.encode_batch(wavs=wave.unsqueeze(0)).squeeze()
                # spk_emb_xvector = speaker_embedding_func_xvector.encode_batch(wavs=wave.unsqueeze(0)).squeeze()
                # combined_utt_condition = torch.cat([spk_emb_ecapa.cpu(), spk_emb_xvector.cpu()], dim=0)

                # embedpath = wavepath.replace('wavs/', 'embeds/').replace('.wav', '.pt')
                # Path(embedpath).parent.mkdir(parents=True, exist_ok=True)
                # torch.save(combined_utt_condition, embedpath)

                line = wavpaths[wavepath].strip().split('|')

                # wavepath = wavepath.replace('wavs/', '16kHz/')
                # Path(wavepath).parent.mkdir(parents=True, exist_ok=True)
                # write(wavepath, SAMPLE_RATE, wave.cpu().numpy())

                # line[0] = wavepath
                # if embedpath not in wavpaths[wavepath]:
                #     line.insert(-1, embedpath)
                text = line[-2]
                if '+' not in text:
                    text = stress_rnn.put_stress(text)
                line[-2] = text
                md.write('|'.join(line) + '\n')


if __name__ == '__main__':
    for md in METADATA_PATHS:
        generate_embeds(md)

# sbatch -t 10-00:00:00 --out preproc.out -G 0 -c 8 --nodelist v100-1 --wrap="python3 -u preprocess.py"
