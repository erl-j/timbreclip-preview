#%%
import torch
from . import model
from pathvalidate import sanitize_filename
import glob


def get_timbreclip_model(clip_version=None, path_to_model=None, device="cpu"):
    # read clip version from path
    if path_to_model is not None:
        sane_clip_version = path_to_model.split("/")[-1].split("_")[1]
    if not path_to_model:
        sane_clip_version = sanitize_filename(clip_version)
        fps = glob.glob(f"./demo_assets/timbreCLIP_{sane_clip_version}*.pt")
        # sort fps
        path_to_model = sorted(fps)[-1]

    z_size = (
        768
        if ("large" in sane_clip_version)
        or ("L14" in sane_clip_version)
        or ("L-14" in sane_clip_version)
        else 512
    )

    print(
        f"loading timbeCLIP with CLIP embedding size {z_size} model from", path_to_model
    )

    timbreclip_model = model.TimbreCLIP(
        embedding_size=z_size, use_wav2clip_architecture="w2carch" in path_to_model
    )
    timbreclip_model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device)))
    return timbreclip_model


# %%
