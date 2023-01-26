#%%
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from transformers import CLIPModel, CLIPProcessor
import timbreCLIP
import glob
import librosa
import torch

SAMPLE_RATE=16000
CLIP_DURATION=4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path ="timbreCLIP_checkpoints/timbreCLIP_laionCLIP-ViT-L-14-laion2B-s32B-b82K_20221110_081911_reinit_w2carch.pt"
timbreclip_model = timbreCLIP.get_model.get_timbreclip_model(path_to_model=checkpoint_path,device=device)

CLIP_MODEL = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

def embed_text(text):
    return clip_model.get_text_features(
        **clip_processor(text, return_tensors="pt", padding=True).to(device)
    ).detach()

# read audio
def load_audio(path, duration):
    x = librosa.load(path, sr=SAMPLE_RATE, duration=duration)[0]
    x = x / (np.max(np.abs(x)) + 1e-7)
    return x
#%% 
audio_paths = glob.glob("test_samples/*.wav")
audio = [load_audio(path,CLIP_DURATION) for path in audio_paths]
audio = torch.tensor(audio).to(device)
audio_z = timbreclip_model(audio)

text = ["church organ","acoustic piano","electric piano"]
text_z = embed_text(text)
#%% show cosine similarity between audio and text
distances = torch.nn.functional.cosine_similarity(text_z[None,:,:],audio_z[:,None,:],dim=-1)
#%% dot product
distances = torch.einsum("ijk,ik->ij",text_z[None,:,:],audio_z)
# show relevance of audio to text
plt.imshow(distances.detach().cpu().numpy())
plt.yticks(range(len(text)),text)
plt.xticks(range(len(audio_paths)),[path.split("/")[-1] for path in audio_paths])
plt.show()
    
    

# %%
