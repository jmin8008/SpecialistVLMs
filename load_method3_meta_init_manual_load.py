"""
방법 3: Meta device init + 수동 load (현재 방식의 최적화)
==========================================================
전략:
  1) Llama3를 from_pretrained 대신 from_config로 빈 껍데기만 생성 (meta 또는 empty)
     → base 가중치 로드를 건너뛰어 메모리/시간 절약
  2) 체크포인트를 dequantize하여 load_state_dict로 덮어쓰기
  3) 체크포인트의 fine-tuned 가중치가 최종 모델에 반영됨

장점: base Llama3 이중 로드 없음 (메모리 절약), 기존 구조 유지
단점: 수동 체크포인트 관리 필요
핵심 차이: 현재 코드는 base Llama3를 먼저 from_pretrained로 로드한 후 덮어쓰지만,
          이 방법은 from_config로 빈 모델만 만들고 체크포인트만 로드한다.
"""

import torch
import json
import os
import glob as glob_module
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel, PretrainedConfig

from models.retinavlm_wrapper import RetinaVLM, RetinaVLMConfig
from models.mini_gpt4 import MiniGPT4
from models.get_model import get_vision_model
from run.vision_language_pretraining import MiniGPT4Module

import hydra


def create_empty_llama3(config):
    """
    Llama3를 from_config로 빈 껍데기만 생성.
    from_pretrained를 호출하지 않으므로 base 가중치를 로드하지 않음.
    → 메모리 절약 + 이중 로드 방지
    """
    print("  [Llama3] from_config으로 빈 모델 생성 (base 가중치 로드 건너뜀)")
    llama_config = AutoConfig.from_pretrained(
        config.model.language_model.model_id,
        cache_dir=config.pretrained_model_dir,
        local_files_only=True,
    )
    # 빈 모델 생성 (가중치는 랜덤 초기화 상태)
    model = AutoModelForCausalLM.from_config(
        llama_config,
        torch_dtype=torch.float16,
    )
    return model


def download_and_dequantize_checkpoint(config):
    """
    HF에서 sharded safetensors를 다운로드하고 int8 → fp16 역양자화.
    """
    repo_id = "RobbieHolland/RetinaVLM"
    subfolder = "RetinaVLM-Specialist"

    # Index 파일 다운로드
    index_path = hf_hub_download(
        repo_id, f"{subfolder}/model.safetensors.index.json",
        cache_dir=config.pretrained_model_dir
    )
    with open(index_path) as f:
        index = json.load(f)

    # Shard 파일들 다운로드 및 병합
    shard_files = set(index["weight_map"].values())
    full_state_dict = {}
    for shard_file in sorted(shard_files):
        print(f"    Loading shard: {shard_file}")
        shard_path = hf_hub_download(
            repo_id, f"{subfolder}/{shard_file}",
            cache_dir=config.pretrained_model_dir
        )
        shard_dict = safetensors_load_file(shard_path)
        full_state_dict.update(shard_dict)

    print(f"    Total keys: {len(full_state_dict)}")

    # int8 + SCB → float16
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
            # SCB is per-row scale: shape (out_features,) → unsqueeze(1) to broadcast over columns
            deq = tensor.float() * scb.float().unsqueeze(1) / 127.0
            dequantized_dict[key] = deq.half()
            n_dequantized += 1
        else:
            dequantized_dict[key] = tensor

    print(f"    Dequantized {n_dequantized} int8 tensors → float16")
    return dequantized_dict


def load_retinavlm_meta_init(config):
    """
    방법 3의 핵심:
    1) Llama3를 빈 껍데기(from_config)로 생성
    2) 전체 모델을 조립
    3) HF 체크포인트를 dequantize하여 load_state_dict
    """
    print("=" * 70)
    print(" 방법 3: Meta/Empty Init + Manual Checkpoint Load")
    print("=" * 70)

    # ── Step 1: Config 준비 ──
    rvlm_config = RetinaVLMConfig.from_pretrained(
        "RobbieHolland/RetinaVLM", subfolder="RetinaVLM-Specialist"
    )
    rvlm_config.update(config)
    rvlm_config.model.checkpoint_path = None

    # Llama3의 initialize를 False로 설정 → from_config만 사용
    # 이렇게 하면 llama3.py에서 from_pretrained 대신 from_config을 호출
    original_initialize = config.model.language_model.initialize

    print("\n  [Step 1] Llama3 initialize=False로 설정")
    print("    → base 가중치 로드 건너뜀 (from_config만 사용)")
    config.model.language_model.initialize = False

    # ── Step 2: 모델 생성 (빈 껍데기) ──
    print("\n  [Step 2] RetinaVLM 빈 모델 생성")
    model = RetinaVLM(rvlm_config)

    # 원래 설정 복원
    config.model.language_model.initialize = original_initialize

    # ── Step 3: 체크포인트 다운로드 및 역양자화 ──
    print("\n  [Step 3] 체크포인트 다운로드 및 역양자화")
    dequantized_dict = download_and_dequantize_checkpoint(config)

    # ── Step 4: 가중치 로드 ──
    print("\n  [Step 4] load_state_dict 실행")

    # 모델의 현재 키와 체크포인트 키 비교
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(dequantized_dict.keys())

    print(f"    모델 키: {len(model_keys)}")
    print(f"    체크포인트 키: {len(ckpt_keys)}")
    print(f"    공통 키: {len(model_keys & ckpt_keys)}")

    missing, unexpected = model.load_state_dict(dequantized_dict, strict=False)

    print(f"\n    LOAD RESULT:")
    print(f"      Loaded: {len(ckpt_keys) - len(unexpected)} keys")
    print(f"      Missing (모델에는 있지만 체크포인트에 없음): {len(missing)}")
    print(f"      Unexpected (체크포인트에는 있지만 모델에 없음): {len(unexpected)}")

    if missing:
        cats = {}
        for k in missing:
            parts = k.split('.')
            cat = parts[1] if len(parts) > 1 else parts[0]
            cats[cat] = cats.get(cat, 0) + 1
        for cat, count in sorted(cats.items()):
            print(f"        Missing [{cat}]: {count}")

    if unexpected:
        cats = {}
        for k in unexpected:
            parts = k.split('.')
            cat = parts[1] if len(parts) > 1 else parts[0]
            cats[cat] = cats.get(cat, 0) + 1
        for cat, count in sorted(cats.items()):
            print(f"        Unexpected [{cat}]: {count}")

    print("\n  로드 완료!")
    return model.eval()


# =====================================================
#  비교 유틸: 현재 방식 vs 방법 3의 가중치 차이
# =====================================================
def compare_with_current_method(config):
    """
    현재 방식(initialize=True → 체크포인트 덮어쓰기)과
    방법 3(initialize=False → 체크포인트만 로드)의 가중치를 비교.
    """
    from models.retinavlm_wrapper import load_retinavlm_specialist_from_hf

    print("\n" + "=" * 70)
    print(" 현재 방식 vs 방법 3 가중치 비교")
    print("=" * 70)

    # 현재 방식
    print("\n[현재 방식] initialize=True + checkpoint overwrite")
    model_current = load_retinavlm_specialist_from_hf(config)

    # 방법 3
    print("\n[방법 3] initialize=False + checkpoint only")
    model_method3 = load_retinavlm_meta_init(config)

    # 비교
    sd_current = model_current.state_dict()
    sd_method3 = model_method3.state_dict()

    common_keys = set(sd_current.keys()) & set(sd_method3.keys())
    print(f"\n  공통 키: {len(common_keys)}")

    n_equal = 0
    n_close = 0
    n_different = 0
    diff_examples = []

    for key in sorted(common_keys):
        t1 = sd_current[key].float()
        t2 = sd_method3[key].float()

        if torch.equal(t1, t2):
            n_equal += 1
        elif torch.allclose(t1, t2, atol=1e-3, rtol=1e-3):
            n_close += 1
        else:
            n_different += 1
            max_diff = (t1 - t2).abs().max().item()
            if len(diff_examples) < 5:
                diff_examples.append((key, max_diff))

    print(f"  완전 동일: {n_equal}")
    print(f"  근사 동일 (atol=1e-3): {n_close}")
    print(f"  다름: {n_different}")

    if diff_examples:
        print(f"\n  다른 키 예시:")
        for key, max_diff in diff_examples:
            print(f"    {key}: max_diff={max_diff:.6f}")


# =====================================================
#  실행
# =====================================================
@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config):
    model = load_retinavlm_meta_init(config)

    # 검증
    print("\n" + "=" * 70)
    print(" 가중치 검증")
    print("=" * 70)
    inner = model.model

    # Visual Encoder
    ve_params = dict(inner.visual_encoder.named_parameters())
    zero_ve = sum(1 for p in ve_params.values() if (p.float() == 0).all().item())
    print(f"  Visual Encoder: {zero_ve}/{len(ve_params)} all-zero params")

    # Projection
    for name, param in inner.llama_proj.named_parameters():
        p = param.float()
        print(f"  llama_proj.{name}: mean={p.mean():.6f}, std={p.std():.6f}")

    # LLM 샘플
    sample_names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.15.mlp.gate_proj.weight",
        "model.layers.31.self_attn.v_proj.weight",
        "lm_head.weight",
    ]
    llm_params = dict(inner.llm.model.named_parameters())
    for name in sample_names:
        if name in llm_params:
            p = llm_params[name].float()
            print(f"  llm.{name}: mean={p.mean():.6f}, std={p.std():.6f}")

    # ── 추론 테스트 ──
    print("\n" + "=" * 70)
    print(" 추론 테스트")
    print("=" * 70)

    DEVICE = next(model.parameters()).device
    print(f"  모델 디바이스: {DEVICE}")

    # 샘플 이미지 로드
    image_dir = "dataset/processed_images"
    image_files = sorted(glob_module.glob(os.path.join(image_dir, "*.png")))
    if not image_files:
        print("  경고: 샘플 이미지를 찾을 수 없습니다.")
        return model

    test_image_path = image_files[0]
    print(f"  테스트 이미지: {os.path.basename(test_image_path)}")

    img = Image.open(test_image_path).convert('L')  # 그레이스케일

    query = "Write an extensive report describing the OCT image and listing any present biomarkers or other observations."
    print(f"  질문: {query}")

    print("\n  추론 시작...")
    with torch.no_grad():
        response = model.forward([img], [query], max_new_tokens=300)

    print(f"\n  모델 응답:")
    print("-" * 70)
    for i, r in enumerate(response):
        print(f"  {r}")
    print("-" * 70)

    return model


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main_compare(config):
    """현재 방식과 방법 3을 비교 실행"""
    compare_with_current_method(config)


if __name__ == "__main__":
    main()
    # main_compare()  # 비교 실행 시 주석 해제
