"""Generate a proper test sample from the validation dataset (same format as training)."""
import sys, yaml, soundfile as sf
import random # added

# cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/tse/miso_tapping.yaml"
# data_dir = sys.argv[2] if len(sys.argv) > 2 else "data/TappingOnly"
cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/tse/miso.yaml"
data_dir = sys.argv[2] if len(sys.argv) > 2 else "data/MisoDataset"

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

from src.tse.train import _build_datasets
_, val_ds = _build_datasets(cfg, data_dir)

# dataset returns (inputs, targets)
#inputs, targets = val_ds[0]

# added 
idx = random.randint(0, len(val_ds) - 1)
inputs, targets = val_ds[idx]


mixture = inputs["mixture"].numpy()    # (2, T) binaural mix
gt      = targets["target"].numpy()    # (1, T) or (T,) clean background
label   = inputs["embedding"]

if gt.ndim == 2:
    gt = gt[0]

sr = cfg["data"].get("sr", 16000)
sf.write("test_mix_binaural.wav", mixture.T, sr)
sf.write("test_gt_background.wav", gt, sr)

print(f"Label: {label.tolist()}")
print(f"Saved test_mix_binaural.wav  ({mixture.shape[1]/sr:.1f}s) — feed this to infer.py")
print(f"Saved test_gt_background.wav — what clean background should sound like")
