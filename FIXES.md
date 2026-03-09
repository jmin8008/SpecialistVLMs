# RetinaVLM 실행 오류 수정 기록

## 목표
`python run/demos/inference.py paths=template` 정상 실행

---

## 수정 1: meta tensor GPU 이동 오류

**파일**: `models/retinavlm_wrapper.py:28`

**에러**:
```
NotImplementedError: Cannot copy out of meta tensor; no data!
Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to()
```

**원인**:
`RetinaVLM.from_pretrained()`이 내부적으로 meta device context를 사용하는데,
그 안에서 빈 vision encoder를 GPU로 옮기려다 실패.

**수정**:
```python
# 전
self.model = MiniGPT4Module(config, device=device).model.eval()

# 후
self.model = MiniGPT4Module(config, device=None).model.eval()
```

---

## 수정 2: Llama3 from_pretrained meta context 충돌

**파일**: `models/llama3.py:33`

**에러**:
```
RuntimeError: You are using `from_pretrained` with a meta device context manager.
This is an anti-pattern as `from_pretrained` wants to load existing weights.
```

**원인**:
`RetinaVLM.from_pretrained()` (meta context) 내부에서
`AutoModelForCausalLM.from_pretrained()` 중복 호출 → 충돌.

**수정**:
```python
# 전
if self.config.model.language_model.initialize:
    self.model = AutoModelForCausalLM.from_pretrained(...)

# 후
is_meta_context = str(torch.get_default_device()) == 'meta'
if self.config.model.language_model.initialize and not is_meta_context:
    self.model = AutoModelForCausalLM.from_pretrained(...)
```

---

## 수정 3: transformers 버전 호환성

**파일**: `models/retinavlm_wrapper.py:19`

**에러**:
```
AttributeError: 'RetinaVLM' object has no attribute 'all_tied_weights_keys'.
Did you mean: '_tied_weights_keys'?
```

**원인**:
transformers 5.3.0이 `_tied_weights_keys` 속성을 기대하는데 `RetinaVLM`에 없음.

**수정**:
```python
class RetinaVLM(PreTrainedModel):
    config_class = RetinaVLMConfig
    _tied_weights_keys = []  # 추가
```

---

## 수정 4: 가중치 키 불일치로 인한 랜덤 출력

**파일**: `models/retinavlm_wrapper.py:92`

**증상**:
모델 실행은 되나 출력이 완전한 쓰레기값 (랜덤 텍스트).

**원인**:
`from_pretrained` 방식의 구조적 한계로 두 핵심 컴포넌트가 랜덤 초기화 상태:

| 컴포넌트 | safetensors 키 | 모델이 기대한 키 | 결과 |
|---------|--------------|--------------|------|
| ResNet50 | `model.visual_encoder.model.*` | `model.visual_encoder.feature_tokens_model.*` | 미로드 |
| Llama3 | `model.llm.model.*` (8bit) | `from_config` 빈 구조 | 미로드 |
| llama_proj | `model.llama_proj.*` | 정상 | 로드됨 |

**수정**:
`from_pretrained` 완전히 우회 → 직접 초기화 후 safetensors에서 명시적 가중치 로드.

---

## 수정 5: Llama3 16GB 재다운로드 문제

**파일**: `models/retinavlm_wrapper.py:92`

**증상**:
수정 4 적용 후 `RetinaVLM(rvlm_config)` 직접 초기화 시
Llama3 8B 모델을 HuggingFace에서 16.1GB 재다운로드 시작.

**원인**:
`rvlm_config.model.language_model.initialize = True` 상태에서
`Llama3.__init__`가 `AutoModelForCausalLM.from_pretrained()` 호출 →
이미 safetensors에 Llama3 가중치가 있음에도 불구하고 새로 다운로드.

**수정**:
`initialize=False`로 설정해 `from_config`로만 빈 구조 생성,
이후 safetensors에서 직접 가중치 로드.

```python
def load_retinavlm_specialist_from_hf(config):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import json, os

    rvlm_config = RetinaVLMConfig.from_pretrained("RobbieHolland/RetinaVLM", subfolder="RetinaVLM-Specialist")
    rvlm_config.update(config)
    rvlm_config.model.checkpoint_path = None
    rvlm_config.model.language_model.initialize = False  # 재다운로드 방지

    # safetensors 먼저 로드
    model_dir = snapshot_download("RobbieHolland/RetinaVLM", subfolder="RetinaVLM-Specialist")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    state_dict = {}
    for shard_file in set(index["weight_map"].values()):
        state_dict.update(load_file(os.path.join(model_dir, shard_file)))

    # 직접 초기화 (initialize=False → from_config, 다운로드 없음)
    model = RetinaVLM(rvlm_config)

    # ResNet50 가중치 로드
    ve_prefix = 'model.visual_encoder.model.'
    ve_state = {k[len(ve_prefix):]: v for k, v in state_dict.items() if k.startswith(ve_prefix)}
    model.model.visual_encoder.model.load_state_dict(ve_state, strict=False)

    # Llama3 가중치 로드 (8bit 메타데이터 키 제외)
    llm_prefix = 'model.llm.model.'
    skip_suffixes = ('weight_format', 'SCB')
    llm_state = {
        k[len(llm_prefix):]: v for k, v in state_dict.items()
        if k.startswith(llm_prefix) and not k.endswith(skip_suffixes)
    }
    model.model.llm.model.load_state_dict(llm_state, strict=False)

    # adapter 가중치 로드
    proj_prefix = 'model.llama_proj.'
    proj_state = {k[len(proj_prefix):]: v for k, v in state_dict.items() if k.startswith(proj_prefix)}
    model.model.llama_proj.load_state_dict(proj_state, strict=True)

    return model.eval()
```
