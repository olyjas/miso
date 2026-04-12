#!/usr/bin/env bash
# Mini Dataset: Download + Evaluate (one-shot pipeline verification)
#
# Downloads the pre-built mini dataset (~250MB) from HuggingFace,
# extracts it, and runs TSE + SED evaluation.
#
# This is the recommended first step for artifact reviewers.
# Expected time: ~30 min on a single GPU.
#
# Usage:
#   bash scripts/eval/eval_mini.sh [data_dir] [output_dir]
#
# Arguments:
#   data_dir    — where to download/extract the mini dataset (default: ./data)
#   output_dir  — where to save evaluation results (default: eval_results/mini)
set -euo pipefail

DATA_DIR="${1:-./data}"
OUTPUT_DIR="${2:-eval_results/mini}"
MINI_TAR="BinauralCuratedDataset_mini.tar.gz"
HF_URL="https://huggingface.co/datasets/ooshyun/fine-grained-soundscape/resolve/main/${MINI_TAR}"

echo "============================================================"
echo "  Mini Dataset: Download + Evaluate"
echo "  Data dir:   ${DATA_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Started:    $(date)"
echo "============================================================"

# ============================================================
# Step 1: Download mini dataset
# ============================================================
mkdir -p "${DATA_DIR}"

if [[ -d "${DATA_DIR}/BinauralCuratedDataset_mini" ]]; then
    echo ""
    echo "[SKIP] Mini dataset already exists at ${DATA_DIR}/BinauralCuratedDataset_mini"
else
    echo ""
    echo "Step 1: Downloading mini dataset (~250 MB)..."
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "${DATA_DIR}/${MINI_TAR}" "${HF_URL}"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "${DATA_DIR}/${MINI_TAR}" "${HF_URL}"
    else
        echo "Error: wget or curl required" >&2
        exit 1
    fi

    echo "Extracting..."
    tar xzf "${DATA_DIR}/${MINI_TAR}" -C "${DATA_DIR}/"
    rm -f "${DATA_DIR}/${MINI_TAR}"
    echo "Done: ${DATA_DIR}/BinauralCuratedDataset_mini/"
fi

# ============================================================
# Step 2: TSE Evaluation (Table 1 subset — 2 models × 2000 samples)
# ============================================================
echo ""
echo "========== Step 2: TSE Evaluation (Table 1) =========="
for model in orange_pi waveformer; do
    echo ""
    echo "--- ${model} --- $(date)"
    python -m src.tse.eval \
        --pretrained ooshyun/fine_grained_soundscape_control \
        --model "${model}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/tse/${model}" \
        --num_samples 2000
done

# ============================================================
# Step 3: SED Evaluation (Fig. 11 subset — tgt=1 and tgt=5)
# ============================================================
echo ""
echo "========== Step 3: SED Evaluation (Fig. 11) =========="
SED_COMMON="--pretrained ooshyun/sound_event_detection --model finetuned_ast \
    --dataset misophonia --root_dataset_dir ${DATA_DIR} \
    --sr 16000 --duration 5 --samples 2000 \
    --num_noise_min 1 --num_noise_max 1 \
    --find_thresholds --val_samples 2000"

echo ""
echo "--- SED tgt=1 (best case) --- $(date)"
python -m src.sed.eval ${SED_COMMON} \
    --num_fg_min 1 --num_fg_max 1 \
    --num_bg_min 1 --num_bg_max 2 \
    --output_dir "${OUTPUT_DIR}/sed/tgt1"

echo ""
echo "--- SED tgt=5 (worst case) --- $(date)"
python -m src.sed.eval ${SED_COMMON} \
    --num_fg_min 5 --num_fg_max 5 \
    --num_bg_min 1 --num_bg_max 2 \
    --output_dir "${OUTPUT_DIR}/sed/tgt5"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  RESULTS — $(date)"
echo "============================================================"

echo ""
echo "=== TSE (Table 1) ==="
echo "  Paper: OrangePi SNRi=11.99/SI-SDRi=11.27, Waveformer SNRi=7.29/SI-SDRi=5.58"
for model in orange_pi waveformer; do
    f="${OUTPUT_DIR}/tse/${model}/metrics_total_averages.json"
    if [[ -f "$f" ]]; then
        python3 -c "
import json; d=json.load(open('$f'))
print(f'  ${model}: SNRi={d[\"snr_i\"]:.2f}, SI-SDRi={d[\"si_sdr_i\"]:.2f}')
"
    else
        echo "  ${model}: N/A"
    fi
done

echo ""
echo "=== SED (Fig. 11) ==="
echo "  Paper: tgt=1 Acc=0.992/F1=0.921, tgt=5 Acc=0.923/F1=0.838"
for d in "${OUTPUT_DIR}"/sed/*/; do
    name=$(basename "$d")
    f="${d}/metrics.json"
    if [[ -f "$f" ]]; then
        python3 -c "
import json; d=json.load(open('$f'))
print(f'  ${name}: Acc={d[\"accuracy\"]:.3f}, Prec={d[\"precision\"]:.3f}, Rec={d[\"recall\"]:.3f}, F1={d[\"f1\"]:.3f}')
"
    else
        echo "  ${name}: N/A"
    fi
done

echo ""
echo "Done! Results saved to ${OUTPUT_DIR}/"
echo "Finished: $(date)"
