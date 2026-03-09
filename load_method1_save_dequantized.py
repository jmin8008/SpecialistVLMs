"""
방법 1: 역양자화된 체크포인트를 미리 저장 후 from_pretrained로 로드
========================================================================
전략:
  1) 현재 방식(수동 다운로드 + dequantize)으로 모델을 한 번 로드
  2) save_pretrained()로 dequantized 가중치를 로컬에 저장
  3) 이후부터는 RetinaVLM.from_pretrained()로 바로 로드
     → meta device init → weight load 자동 처리

장점: 가장 깔끔. 이후 로드가 간단하고, HF 표준 파이프라인 그대로 사용
단점: 한 번은 기존 방식으로 로드해야 하고, 디스크 공간 추가 사용 (~16GB fp16)
"""

import torch
import json
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file
from transformers import PreTrainedModel, PretrainedConfig

from models.retinavlm_wrapper import RetinaVLM, RetinaVLMConfig
from run.vision_language_pretraining import MiniGPT4Module

import hydra
from omegaconf import OmegaConf


# =====================================================
#  Step 1: dequantized 체크포인트 저장 (한 번만 실행)
# =====================================================
def save_dequantized_checkpoint(config, save_dir="saved_models/RetinaVLM-Specialist-Dequantized"):
    """
    HF에서 8-bit 체크포인트를 받아 dequantize한 후,
    from_pretrained 호환 형태로 로컬에 저장한다.
    """
    print("=" * 70)
    print(" Step 1: 기존 방식으로 모델 로드 (수동 dequantization)")
    print("=" * 70)

    # 1a. Config 로드
    rvlm_config = RetinaVLMConfig.from_pretrained(
        "RobbieHolland/RetinaVLM", subfolder="RetinaVLM-Specialist"
    )
    rvlm_config.update(config)
    rvlm_config.model.checkpoint_path = None

    # 1b. 모델 직접 생성 (내부에서 base Llama3 로드됨)
    print("Creating RetinaVLM model...")
    model = RetinaVLM(rvlm_config)

    # 1c. HF에서 sharded safetensors 다운로드
    print("Downloading checkpoint from HuggingFace...")
    repo_id = "RobbieHolland/RetinaVLM"
    subfolder = "RetinaVLM-Specialist"

    index_path = hf_hub_download(
        repo_id, f"{subfolder}/model.safetensors.index.json",
        cache_dir=config.pretrained_model_dir
    )
    with open(index_path) as f:
        index = json.load(f)

    shard_files = set(index["weight_map"].values())
    full_state_dict = {}
    for shard_file in sorted(shard_files):
        print(f"  Loading shard: {shard_file}")
        shard_path = hf_hub_download(
            repo_id, f"{subfolder}/{shard_file}",
            cache_dir=config.pretrained_model_dir
        )
        shard_dict = safetensors_load_file(shard_path)
        full_state_dict.update(shard_dict)

    print(f"  Total keys in checkpoint: {len(full_state_dict)}")

    # 1d. 8-bit → float16 역양자화
    dequantized_dict = {}
    scb_keys = {k for k in full_state_dict if k.endswith('.SCB')}
    weight_format_keys = {k for k in full_state_dict if k.endswith('.weight_format')}
    skip_keys = scb_keys | weight_format_keys

    n_dequantized = 0
    for key, tensor in full_state_dict.items():
        if key in skip_keys:
            continue

        scb_key = key.rsplit('.', 1)[0] + '.SCB' if '.' in key else key + '.SCB'
        if tensor.dtype == torch.int8 and scb_key in full_state_dict:
            scb = full_state_dict[scb_key]
            dequantized = tensor.float() * scb.float().unsqueeze(1) / 127.0
            dequantized_dict[key] = dequantized.half()
            n_dequantized += 1
        else:
            dequantized_dict[key] = tensor

    print(f"  Dequantized {n_dequantized} int8 tensors to float16")

    # 1e. 모델에 가중치 적용
    missing, unexpected = model.load_state_dict(dequantized_dict, strict=False)
    print(f"  Loaded: {len(dequantized_dict) - len(unexpected)} keys")
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # 1f. dequantized 모델을 HF 호환 형태로 저장
    print(f"\n  Saving dequantized model to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 수동 저장: torch.save (tied weights 허용) + config JSON
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(save_dir, "model.pt"))
    print(f"  Saved {len(state_dict)} tensors")

    # config를 JSON 직렬화 가능한 형태로 변환 후 저장
    config_dict = {}
    for attr_name in list(vars(model.config)):
        if attr_name.startswith('_'):
            continue
        val = getattr(model.config, attr_name)
        try:
            json.dumps(val)
            config_dict[attr_name] = val
        except (TypeError, ValueError):
            if hasattr(val, '__class__') and ('DictConfig' in val.__class__.__name__ or 'ListConfig' in val.__class__.__name__):
                config_dict[attr_name] = OmegaConf.to_container(val, resolve=True)

    config_dict["model_type"] = "retinavlm"
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Done! 저장 완료.")

    return save_dir


# =====================================================
#  Step 2: from_pretrained로 바로 로드 (이후 매번 사용)
# =====================================================
def load_from_dequantized(config, save_dir="saved_models/RetinaVLM-Specialist-Dequantized"):
    """
    미리 저장해둔 dequantized 체크포인트에서 로드.
    모델을 새로 생성한 뒤, torch.load로 state_dict를 적용.
    """
    print("=" * 70)
    print(" Step 2: dequantized 모델 로드")
    print("=" * 70)

    # Config 로드
    rvlm_config = RetinaVLMConfig.from_pretrained(
        "RobbieHolland/RetinaVLM", subfolder="RetinaVLM-Specialist"
    )
    rvlm_config.update(config)
    rvlm_config.model.checkpoint_path = None

    # 모델 생성
    print("  Creating model...")
    model = RetinaVLM(rvlm_config)

    # 저장된 state_dict 로드
    print("  Loading saved state_dict...")
    state_dict = torch.load(os.path.join(save_dir, "model.pt"), map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded: {len(state_dict) - len(unexpected)} keys")
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model.eval()
    print("  로드 완료!")
    return model


# =====================================================
#  실행
# =====================================================
@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config):
    save_dir = "saved_models/RetinaVLM-Specialist-Dequantized"

    if not os.path.exists(save_dir):
        # 처음: dequantized 체크포인트 생성 및 저장
        print("[첫 실행] Dequantized 체크포인트를 생성합니다...")
        save_dequantized_checkpoint(config, save_dir)
    else:
        print(f"[이미 존재] {save_dir}")

    # from_pretrained로 로드
    model = load_from_dequantized(config, save_dir)

    # 검증: 가중치 통계
    print("\n" + "=" * 70)
    print(" 가중치 검증")
    print("=" * 70)
    inner = model.model  # MiniGPT4 instance

    # Visual Encoder
    ve_params = dict(inner.visual_encoder.named_parameters())
    zero_ve = sum(1 for p in ve_params.values() if (p.float() == 0).all().item())
    print(f"  Visual Encoder: {zero_ve}/{len(ve_params)} all-zero params")

    # Projection
    for name, param in inner.llama_proj.named_parameters():
        p = param.float()
        print(f"  llama_proj.{name}: mean={p.mean():.6f}, std={p.std():.6f}")

    # LLM 샘플
    for name, param in list(inner.llm.model.named_parameters())[:3]:
        p = param.float()
        print(f"  llm.{name}: mean={p.mean():.6f}, std={p.std():.6f}")

    # =====================================================
    #  추론 테스트
    # =====================================================
    print("\n" + "=" * 70)
    print(" 추론 테스트")
    print("=" * 70)

    from PIL import Image
    from glob import glob
    import numpy as np

    # GPU로 이동 (가능하면)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")

    # 샘플 이미지 로드
    sample_dir = "dataset/processed_images"
    image_files = sorted(glob(os.path.join(sample_dir, "*.png")))
    if not image_files:
        print("  샘플 이미지가 없습니다!")
        return model

    image_path = image_files[0]
    image = np.array(Image.open(image_path).convert('L'))
    print(f"  Image: {os.path.basename(image_path)}")
    print(f"  Shape: {image.shape}")

    # 질문 & 추론
    query = "Write a detailed clinical report describing this OCT scan. Identify any visible biomarkers such as drusen, fluid, or atrophy."
    print(f"  Query: {query}\n")

    with torch.no_grad():
        outputs = model.forward([image], [query], max_new_tokens=300)

    print("  [AI Response]")
    print(f"  {outputs[0]}")

    return model


if __name__ == "__main__":
    main()
