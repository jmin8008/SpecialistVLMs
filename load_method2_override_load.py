"""
방법 2: from_pretrained 내부 오버라이드를 통한 자동 dequantize + state_dict 로드
=================================================================================
전략:
  1) RetinaVLM 모델을 직접 생성 (initialize=False로 빈 LLM)
  2) HF 체크포인트를 다운로드하여 int8 → fp16 역양자화
  3) 키 리매핑 후 load_state_dict로 가중치 로드

장점: 별도 dequantized 체크포인트 저장 불필요, HF에서 직접 로드
단점: 키 리매핑 로직이 모델 구조에 의존
"""

import torch
import json
import os
from typing import Optional, List, Tuple
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file as safetensors_load_file

from models.retinavlm_wrapper import RetinaVLMConfig
from run.vision_language_pretraining import MiniGPT4Module
from models.mini_gpt4 import MiniGPT4

import hydra


def dequantize_state_dict(state_dict):
    """
    int8 + SCB 키를 찾아 float16으로 역양자화한다.
    HF 체크포인트의 8-bit 양자화 가중치를 처리.
    """
    dequantized = {}
    scb_keys = {k for k in state_dict if k.endswith('.SCB')}
    weight_format_keys = {k for k in state_dict if k.endswith('.weight_format')}
    skip_keys = scb_keys | weight_format_keys
    n_dequantized = 0

    for key, tensor in state_dict.items():
        if key in skip_keys:
            continue

        scb_key = key.rsplit('.', 1)[0] + '.SCB' if '.' in key else key + '.SCB'
        if tensor.dtype == torch.int8 and scb_key in state_dict:
            scb = state_dict[scb_key]
            w = tensor.float()
            s = scb.float()
            if w.ndim == 2:
                if s.shape[0] == w.shape[1]:
                    deq = w * s.unsqueeze(0) / 127.0
                elif s.shape[0] == w.shape[0]:
                    deq = w * s.unsqueeze(1) / 127.0
                else:
                    print(f"    WARNING: SCB shape {s.shape} does not match weight shape {w.shape}, skipping dequant for {key}")
                    dequantized[key] = tensor
                    continue
            else:
                deq = w * s / 127.0
            dequantized[key] = deq.half()
            n_dequantized += 1
        else:
            dequantized[key] = tensor

    print(f"  [dequantize_state_dict] {n_dequantized} int8 tensors → float16")
    return dequantized


def load_hf_checkpoint_files(repo_id, subfolder, cache_dir=None):
    """
    HF 체크포인트 파일들을 다운로드하고 합쳐서 state_dict로 반환.
    """
    # 모델 파일 다운로드 (snapshot_download로 전체 subfolder)
    local_dir = snapshot_download(
        repo_id,
        allow_patterns=[f"{subfolder}/*.safetensors", f"{subfolder}/*.bin", f"{subfolder}/model.safetensors.index.json"],
        cache_dir=cache_dir,
    )

    model_dir = os.path.join(local_dir, subfolder)

    # index 파일 확인 (sharded checkpoint)
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        print(f"  [load_hf_checkpoint] Found {len(shard_files)} sharded safetensors files")
    else:
        # 단일 파일
        shard_files = [f for f in os.listdir(model_dir)
                       if f.endswith('.safetensors') or f.endswith('.bin')]
        print(f"  [load_hf_checkpoint] Found {len(shard_files)} checkpoint files")

    full_state_dict = {}
    for shard_file in sorted(shard_files):
        filepath = os.path.join(model_dir, shard_file)
        if filepath.endswith('.safetensors'):
            shard = safetensors_load_file(filepath)
        else:
            shard = torch.load(filepath, map_location="cpu")
            if 'state_dict' in shard:
                shard = shard['state_dict']
        full_state_dict.update(shard)
        print(f"    Loaded {shard_file}: {len(shard)} tensors")

    print(f"  [load_hf_checkpoint] Total: {len(full_state_dict)} tensors")
    return full_state_dict


def remap_checkpoint_keys(ckpt_state_dict, model_state_dict):
    """
    체크포인트 키를 모델 키에 맞게 리매핑.
    1) "model." prefix 제거 (체크포인트는 RetinaVLM wrapper 기준, 직접 로드는 MiniGPT4 기준)
    2) llm.model.model.X → llama_model.model.X 변환
    3) 중복 키 생성 (llama_model ↔ llm.model, MiniGPT4의 alias 구조 대응)
    """
    expected_keys = set(model_state_dict.keys())

    # Step 1: "model." prefix 제거
    stripped = {}
    for k, v in ckpt_state_dict.items():
        new_k = k[len("model."):] if k.startswith("model.") else k
        stripped[new_k] = v

    # Step 2: llm.model → llama_model 변환
    remap_rules = [
        ("llm.model.model.", "llama_model.model."),
        ("llm.model.lm_head.", "llama_model.lm_head."),
    ]
    remapped = {}
    for k, v in stripped.items():
        new_k = k
        for old_pattern, new_pattern in remap_rules:
            new_k = new_k.replace(old_pattern, new_pattern)
        remapped[new_k] = v

    # Step 3: 중복 키 생성 (alias 구조 대응)
    # MiniGPT4: self.llama_model = self.llm.model
    # Visual encoder: self.feature_tokens_model = self.model (or similar)
    alias_pairs = [
        ("llama_model.model.", "llm.model.model."),
        ("llama_model.lm_head.", "llm.model.lm_head."),
    ]
    extra = {}
    for k, v in remapped.items():
        for prefix_a, prefix_b in alias_pairs:
            if k.startswith(prefix_a):
                alt_k = prefix_b + k[len(prefix_a):]
                if alt_k not in remapped:
                    extra[alt_k] = v
            elif k.startswith(prefix_b):
                alt_k = prefix_a + k[len(prefix_b):]
                if alt_k not in remapped:
                    extra[alt_k] = v
    remapped.update(extra)

    # Step 4: visual_encoder.model.X → visual_encoder.feature_tokens_model.N.X 매핑
    # PretrainedResNet: feature_tokens_model = Sequential(*list(model.children())[:-1])
    # ResNet children 순서: conv1→0, bn1→1, relu→2, maxpool→3, layer1→4, ...layer4→7
    resnet_name_to_idx = {
        "conv1": "0", "bn1": "1", "relu": "2", "maxpool": "3",
        "layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7",
    }
    ve_extra = {}
    for k, v in remapped.items():
        if k.startswith("visual_encoder.model."):
            suffix = k[len("visual_encoder.model."):]
            parts = suffix.split(".", 1)
            child_name = parts[0]
            if child_name in resnet_name_to_idx:
                idx = resnet_name_to_idx[child_name]
                rest = "." + parts[1] if len(parts) > 1 else ""
                ftm_key = f"visual_encoder.feature_tokens_model.{idx}{rest}"
                if ftm_key not in remapped:
                    ve_extra[ftm_key] = v
    remapped.update(ve_extra)

    # 결과 확인
    remapped_keys = set(remapped.keys())
    matched = expected_keys & remapped_keys
    missing = expected_keys - remapped_keys
    unexpected = remapped_keys - expected_keys

    print(f"  [remap] Matched={len(matched)}/{len(expected_keys)}, Missing={len(missing)}, Unexpected={len(unexpected)}")

    if missing:
        print(f"  [remap] Sample missing keys (first 5): {sorted(missing)[:5]}")
    if unexpected:
        print(f"  [remap] Sample unexpected keys (first 5): {sorted(unexpected)[:5]}")

    return remapped


# =====================================================
#  로드 함수
# =====================================================
def load_retinavlm_with_override(config):
    """
    1) 모델을 직접 생성 (빈 LLM)
    2) HF 체크포인트를 다운로드 + 역양자화
    3) load_state_dict로 가중치 로드
    """
    print("=" * 70)
    print(" 방법 2: 직접 모델 생성 + HF 체크포인트 dequantize 로드")
    print("=" * 70)

    # Config 로드
    rvlm_config = RetinaVLMConfig.from_pretrained(
        "RobbieHolland/RetinaVLM", subfolder="RetinaVLM-Specialist"
    )
    rvlm_config.update(config)
    rvlm_config.model.checkpoint_path = None
    # LLM은 사전학습 가중치로 초기화 (from_pretrained)
    # 이후 RetinaVLM 체크포인트의 fine-tuned 가중치로 덮어씀
    rvlm_config.model.language_model.initialize = True

    # Step 1: 모델 생성 (Llama3 사전학습 가중치 포함)
    print("\n[Step 1] 모델 구조 생성 (Llama3 사전학습 가중치 로드)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniGPT4Module(rvlm_config, device=device).model.eval()

    # Step 2: HF 체크포인트 다운로드 + 역양자화
    print("\n[Step 2] HF 체크포인트 다운로드 및 역양자화...")
    ckpt_state_dict = load_hf_checkpoint_files(
        "RobbieHolland/RetinaVLM",
        "RetinaVLM-Specialist",
        cache_dir=config.pretrained_model_dir,
    )
    ckpt_state_dict = dequantize_state_dict(ckpt_state_dict)

    # Step 3: 키 리매핑 + 로드
    print("\n[Step 3] 키 리매핑 및 가중치 로드...")
    model_state_dict = model.state_dict()

    # 모델 키에서 중복 감지 (llm.model vs llama_model)
    print(f"  Model state_dict keys: {len(model_state_dict)}")
    print(f"  Checkpoint keys: {len(ckpt_state_dict)}")
    print(f"  Model keys (first 5): {sorted(model_state_dict.keys())[:5]}")
    print(f"  Checkpoint keys (first 5): {sorted(ckpt_state_dict.keys())[:5]}")

    remapped_dict = remap_checkpoint_keys(ckpt_state_dict, model_state_dict)

    # RetinaVLM 학습 시 LLM은 frozen → 체크포인트의 LLM 가중치(int8 역양자화)는
    # 원본보다 품질이 낮으므로 제외하고, 사전학습 가중치를 유지
    llm_prefixes = ("llm.", "llama_model.")
    filtered_dict = {k: v for k, v in remapped_dict.items()
                     if not any(k.startswith(p) for p in llm_prefixes)}
    print(f"  Filtered: {len(remapped_dict)} → {len(filtered_dict)} (LLM 가중치 {len(remapped_dict) - len(filtered_dict)}개 제외)")

    # load_state_dict (strict=False로 missing/unexpected 허용)
    result = model.load_state_dict(filtered_dict, strict=False)

    n_missing = len(result.missing_keys)
    n_unexpected = len(result.unexpected_keys)
    print(f"\n  load_state_dict result: {n_missing} missing, {n_unexpected} unexpected")

    if n_missing > 0:
        # 카테고리별로 missing keys 분류
        categories = {}
        for k in result.missing_keys:
            parts = k.split('.')
            cat = '.'.join(parts[:2]) if len(parts) > 2 else k
            categories[cat] = categories.get(cat, 0) + 1
        print(f"  Missing keys by category:")
        for cat, count in sorted(categories.items()):
            print(f"    {cat}: {count}")

    if n_unexpected > 0:
        print(f"  Unexpected keys (first 10): {result.unexpected_keys[:10]}")

    model = model.to(device)
    print("\n  로드 완료!")
    return model


# =====================================================
#  실행
# =====================================================
@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config):
    model = load_retinavlm_with_override(config)

    # 검증
    print("\n" + "=" * 70)
    print(" 가중치 검증")
    print("=" * 70)

    ve_params = dict(model.visual_encoder.named_parameters())
    zero_ve = sum(1 for p in ve_params.values() if (p.float() == 0).all().item())
    print(f"  Visual Encoder: {zero_ve}/{len(ve_params)} all-zero params")

    for name, param in model.llama_proj.named_parameters():
        p = param.float()
        print(f"  llama_proj.{name}: mean={p.mean():.6f}, std={p.std():.6f}")

    for name, param in list(model.llm.model.named_parameters())[:3]:
        p = param.float()
        print(f"  llm.{name}: mean={p.mean():.6f}, std={p.std():.6f}")

    # =====================================================
    #  추론 테스트
    # =====================================================
    print("\n" + "=" * 70)
    print(" 추론 테스트")
    print("=" * 70)

    from PIL import Image
    import numpy as np
    import scipy.ndimage

    # OCT 이미지 로드 및 전처리
    img_path = "dataset/processed_images/Normal-macular-OCT-1.png"
    print(f"  이미지 로드: {img_path}")
    img = Image.open(img_path)
    img_np = np.array(img)

    # 채널 처리
    if img_np.ndim == 3:
        if img_np.shape[2] == 4:  # RGBA → RGB
            img_np = img_np[:, :, :3]
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC → CHW
    elif img_np.ndim == 2:
        img_np = img_np[np.newaxis, :, :]  # Grayscale → 1xHxW

    # 192x192 리사이즈
    if img_np.shape[1] != 192 or img_np.shape[2] != 192:
        zoom_factors = [1, 192 / img_np.shape[1], 192 / img_np.shape[2]]
        img_np = scipy.ndimage.zoom(img_np, zoom_factors, order=1)

    # 정규화 (0-255 → 0-1)
    if not np.issubdtype(img_np.dtype, np.floating):
        img_np = img_np.astype(np.float32) / 255.0

    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, C, 192, 192]

    model_param = next(model.parameters())
    img_tensor = img_tensor.to(device=model_param.device, dtype=model_param.dtype)
    print(f"  이미지 텐서 shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")

    # LLM 단독 테스트 (이미지 없이)
    print("\n  --- LLM 단독 텍스트 생성 테스트 ---")
    tokenizer = model.llm.tokenizer
    test_prompt = "The capital of France is"
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(model_param.device)
    input_embeds = model.llama_model.model.embed_tokens(input_ids)
    input_embeds = input_embeds.to(dtype=model_param.dtype)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        llm_out = model.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=30,
            do_sample=False,
        )
    llm_text = tokenizer.decode(llm_out[0], skip_special_tokens=True)
    print(f"  프롬프트: '{test_prompt}'")
    print(f"  LLM 출력: '{llm_text}'")

    # 이미지 + 질의 추론
    queries = ["Describe this retinal OCT image. List any biomarkers or abnormalities you observe."]
    print(f"\n  --- 이미지 + 질의 추론 ---")
    print(f"  질의: {queries[0]}")

    with torch.no_grad():
        outputs = model.query(
            img_tensor,
            queries,
            answer_preamble=[''],
            max_new_tokens=200,
            output_only=True,
            return_samples=False,
        )

    print(f"\n  === 모델 응답 ===")
    for i, resp in enumerate(outputs):
        print(f"  {resp}")

    return model


if __name__ == "__main__":
    main()
