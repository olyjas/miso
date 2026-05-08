"""TSE model wrapper.

Provides a ``load_pretrained()`` helper for checkpoint loading from
HuggingFace Hub.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

# Re-export model classes
from src.tse.net import Net

logger = logging.getLogger(__name__)


# Alias for convenience — Net is the top-level STFT wrapper (TFGridNet)
TFGridNet = Net


# ---------------------------------------------------------------------------
# Pretrained model loading
# ---------------------------------------------------------------------------

_MODEL_NAME_MAP = {
    # Table 1: TSE model comparison (1ch, 1out)
    "orange_pi": "tfgridnet_large_snr_ctl_v2_1ch_1spk_1out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight",
    "raspberry_pi": "tfgridnet_small_snr_ctl_v2_1ch_1spk_1out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight",
    "neuralaid": "tfmlpnet_snr_ctl_v2_1ch_1spk_1out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight",
    "waveformer": "waveformer_snr_ctl_v2_1ch_1spk_1out_20000samples_20sounds_16000sr_256chunk_film_all_onflight",
    # Table 2: Multi-output (Orange Pi)
    "orange_pi_5out": "tfgridnet_large_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight",
    "orange_pi_20out": "tfgridnet_large_snr_ctl_v2_20ch_5spk_20out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight",
    # Table 3: FiLM ablation — Orange Pi
    "orange_pi_film_first": "tfgridnet_large_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_first_onflight",
    "orange_pi_film_all": "tfgridnet_large_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_all_onflight",
    "orange_pi_film_all_except_first": "tfgridnet_large_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight",
    # Table 3: FiLM ablation — NeuralAids
    "neuralaid_film_first": "tfmlpnet_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_first_onflight",
    "neuralaid_film_all": "tfmlpnet_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_all_layers_6_onflight",
    "neuralaid_film_all_except_first": "tfmlpnet_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight",
    # Legacy aliases
    "orange_pi_5ch": "tfgridnet_large_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_all_except_first_onflight",
    "orange_pi_5ch_film_all": "tfgridnet_large_snr_ctl_v2_5ch_5spk_5out_20000samples_20sounds_16000sr_96chunk_film_all_onflight",
}


def load_pretrained(
    repo_id: str = "ooshyun/fine_grained_soundscape_control",
    model_name: str = "orange_pi",
) -> Net:
    """Download and instantiate a pretrained :class:`Net` (TFGridNet) from HuggingFace.

    Args:
        repo_id: HuggingFace Hub repository ID.
        model_name: One of ``"orange_pi"``, ``"raspberry_pi"``, ``"neuralaid"``.

    Returns:
        A :class:`Net` with pretrained weights loaded.
    """
    import json

    from huggingface_hub import hf_hub_download

    if model_name not in _MODEL_NAME_MAP:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Available: {list(_MODEL_NAME_MAP.keys())}"
        )

    run_dir = _MODEL_NAME_MAP[model_name]

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{run_dir}/config.json",
    )
    # Try last.pt first (paper uses last epoch), fallback to best.pt
    try:
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{run_dir}/checkpoints/last.pt",
        )
    except Exception:
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{run_dir}/checkpoints/best.pt",
        )

    with open(config_path) as f:
        config = json.load(f)

    # Build model from config — handle nested pl_module_args.model_params
    if "pl_module_args" in config and "model_params" in config["pl_module_args"]:
        mp = config["pl_module_args"]["model_params"]
    elif "model" in config:
        mp = config["model"]
    else:
        mp = config

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Handle wrapped state dicts (e.g. {"model": ...})
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # Waveformer: uses SemanticHearing DCC-TF architecture
    if "waveformer_params" in mp:
        model = _load_waveformer(mp, state_dict)
        logger.info("Loaded pretrained Waveformer '%s' from %s", model_name, repo_id)
        return model

    # TFGridNet / TFMLPNet: remap original import paths
    _IMPORT_REMAP = {
        "src.models.GuidedTFNetwork.multiflim_guided_tfnet.MultiFiLMGuidedTFNet": "src.tse.multiflim_guided_tfnet.MultiFiLMGuidedTFNet",
        "src.models.blocks.gridnet_blockTFGridNet.GridNetBlock": "src.tse.gridnet_block.GridNetBlock",
        "src.models.blocks.mlpnet_block.MLPBlock": "src.tse.mlpnet_block.MLPBlock",
    }
    raw_model_name = mp.get(
        "model_name", "src.tse.multiflim_guided_tfnet.MultiFiLMGuidedTFNet"
    )
    raw_block_name = mp.get("block_model_name", "src.tse.gridnet_block.GridNetBlock")
    model_name_dotted = _IMPORT_REMAP.get(raw_model_name, raw_model_name)
    block_model_name = _IMPORT_REMAP.get(raw_block_name, raw_block_name)
    block_model_params = mp.get("block_model_params", {})
    embedding_params = mp.get(
        "embedding_params",
        {
            "embedding_dim": 0,
            "embedding_type": "",
            "embedding_activation": "",
            "embedding_init": "",
        },
    )
    film_params = mp.get(
        "film_params",
        {
            "film_positions": [],
            "film_preset": "all_except_first",
        },
    )

    model = Net(
        model_name=model_name_dotted,
        block_model_name=block_model_name,
        block_model_params=block_model_params,
        speaker_dim=mp.get("speaker_dim", 20),
        stft_chunk_size=mp.get("stft_chunk_size", 96),
        stft_pad_size=mp.get("stft_pad_size", 64),
        stft_back_pad=mp.get("stft_back_pad", 96),
        num_input_channels=mp.get("num_input_channels", 1),
        num_output_channels=mp.get("num_output_channels", 1),
        num_layers=mp.get("num_layers", 6),
        latent_dim=mp.get("latent_dim", 32),
        embedding_params=embedding_params,
        film_params=film_params,
        use_first_ln=mp.get("use_first_ln", False),
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    logger.info("Loaded pretrained model '%s' from %s", model_name, repo_id)
    return model


def _load_waveformer(mp: dict, state_dict: dict) -> Any:
    """Load Waveformer (SemanticHearing DCC-TF) from config + state dict."""
    from src.tse.dcc_tf import Net as WaveformerNet

    wp = mp["waveformer_params"]
    in_ch = wp.get("in_channels", 2)
    out_ch = wp.get("out_channels", 2)
    L = wp.get("L", 32)
    model_dim = wp.get("model_dim", 256)
    out_buf_len = wp.get("out_buf_len", 4)

    model = WaveformerNet(
        label_len=wp.get("label_len", 20),
        L=L,
        model_dim=model_dim,
        num_enc_layers=wp.get("num_enc_layers", 10),
        dec_buf_len=wp.get("dec_buf_len", 13),
        num_dec_layers=wp.get("num_dec_layers", 1),
        dec_chunk_size=wp.get("dec_chunk_size", 13),
        out_buf_len=out_buf_len,
        use_pos_enc=wp.get("use_pos_enc", True),
        conditioning=wp.get("conditioning", "mult"),
    )

    # Patch in_conv / out_conv for multi-channel (dcc_tf.Net defaults to 1ch)
    if in_ch != 1:
        lookahead = True  # default
        kernel_size = 3 * L if lookahead else L
        model.in_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_ch, model_dim, kernel_size, stride=L, padding=0, bias=False
            ),
            torch.nn.ReLU(),
        )
    if out_ch != 1:
        model.out_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(
                model_dim,
                out_ch,
                kernel_size=(out_buf_len + 1) * L,
                stride=L,
                padding=out_buf_len * L,
                bias=False,
            ),
            torch.nn.Tanh(),
        )

    # State dict has "model." prefix from parent wrapper — strip it
    stripped = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
        stripped[new_key] = v

    model.load_state_dict(stripped, strict=False)
    model.eval()

    # Wrap in adapter that matches eval.py's dict input/output interface
    # nO=1: Waveformer extracts one source at a time (stereo→mono inside forward).
    # eval.py uses nO==1 to trigger per-label inference for multi-source mixtures.
    wrapper = _WaveformerWrapper(
        model, nI=in_ch, nO=1, label_len=wp.get("label_len", 20)
    )
    return wrapper


class _WaveformerWrapper(torch.nn.Module):
    """Adapter: dict-based forward interface for Waveformer (dcc_tf.Net)."""

    def __init__(self, model, nI=2, nO=2, label_len=20):
        super().__init__()
        self.model = model
        self.nI = nI
        self.nO = nO
        self.label_len = label_len

    def forward(self, inputs):
        x = inputs["mixture"]  # (B, nI, T)

        # Get label vector
        if "embedding" in inputs and inputs["embedding"] is not None:
            label = inputs["embedding"]
        elif "label_vector" in inputs:
            label = inputs["label_vector"]
        else:
            raise ValueError("Need 'embedding' or 'label_vector' in inputs")

        if label.dim() == 1:
            label = label.unsqueeze(0)

        # Init buffers
        batch_size = x.shape[0]
        enc_buf, dec_buf, out_buf = self.model.init_buffers(batch_size, x.device)

        # Waveformer predict — processes full sequence
        output, _, _, _ = self.model.predict(x, label, enc_buf, dec_buf, out_buf)

        # Waveformer outputs stereo (B, 2, T) — convert to mono (B, 1, T)
        if output.shape[1] == 2:
            output = output.mean(dim=1, keepdim=True)

        return {"output": output}
