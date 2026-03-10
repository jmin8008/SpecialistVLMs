# RetinaVLM 비트 및 정밀도(Precision) 관련 분석 보고서

본 보고서는 RetinaVLM 모델을 GPU 환경에서 추론 및 실험할 때 발생할 수 있는 수치적 안정성 문제와 비트 정밀도 관련 이슈를 정리한 것입니다.

## 1. 주요 정밀도 이슈 요약

### 1.1 데이터 타입 불일치 (Dtype Mismatch)
*   **현상**: `RuntimeError: expected scalar type Float but found Half` 등의 오류 발생.
*   **원인**: 
    *   **Vision Encoder**: 보통 `float32`로 특징을 추출함.
    *   **LLM (Llama 3)**: 메모리 효율을 위해 `float16` 또는 `bfloat16`으로 로드됨.
    *   **Projection Layer**: 두 모델 사이의 가중치가 서로 다른 정밀도를 가질 때 연산 충돌 발생.
*   **해결**: 입력 이미지 텐서를 GPU로 보낼 때 모델의 가중치 타입(`model.dtype`)과 일치하도록 강제 캐스팅(Casting) 로직 적용.

### 1.2 양자화(Quantization)에 따른 정보 손실
*   **현상**: AI가 미세한 병변(예: 작은 드루젠, 망막하액)을 감지하지 못하거나 문법에 맞지 않는 답변 생성.
*   **원인**: Llama 3를 8-bit 또는 4-bit로 양자화할 경우, 매우 정밀한 수치인 시각 토큰(Vision Tokens)이 압축된 언어 공간으로 투영되면서 미세한 대조도(Contrast) 정보가 뭉개짐.
*   **대책**: 의료 영상 분석의 특성상 가능하면 **투영 레이어(Projection Layer)는 `float32` 또는 `bfloat16` 정밀도를 유지**할 것을 권장함.

### 1.3 입력 이미지 비트 심도 (8-bit vs 16-bit)
*   **현상**: 이미지가 너무 어둡거나(Black out) 너무 밝게(Saturated) 분석되어 진단 오류 발생.
*   **원인**: OCT 원본 데이터는 16-bit인 경우가 많으나, 코드가 이를 8-bit(0-255)로 가정하고 `255.0`으로 나누어 정규화할 경우 수치 범위가 맞지 않음.
*   **대책**: 입력 데이터의 실제 비트 심도를 확인하거나, 최대값 기준 정규화(`img / img.max()`) 로직 검토 필요.

## 2. 하드웨어별 최적 설정

| GPU 아키텍처 | 권장 정밀도 (Dtype) | 비고 |
| :--- | :--- | :--- |
| **NVIDIA Ampere 이상** (RTX 30/40, A100 등) | `bfloat16` | Llama 3 최적화 타입, 수치 안정성 높음 |
| **NVIDIA Turing/Pascal** (RTX 20, GTX 10 등) | `float16` | `bfloat16` 미지원, 연산 속도 빠름 |
| **CPU / 구형 GPU** | `float32` | 메모리 점유는 높으나 가장 안전함 |

## 3. 코드 적용 완료 사항 (v1.1)

현재 `models/retinavlm_wrapper.py`와 `retfound_projection_llm_experiment.ipynb`에 다음의 안전 장치가 적용되었습니다:

1.  **동적 캐스팅**: 모델이 `float16`이든 `bfloat16`이든, 입력되는 이미지 텐서가 자동으로 모델의 정밀도에 맞춰 변환되도록 수정됨.
2.  **AttributeError 수정**: 입력 데이터가 리스트(List) 형태일 때 `.shape` 접근으로 인한 런타임 에러 방지 로직 추가.
3.  **추론 최적화**: 불필요한 그래디언트 계산을 방지하기 위해 `torch.no_grad()` 환경에서 실행되도록 구성.

---

## 4. Dequantized Inference 성공까지의 문제 해결 과정 (v1.2)

`load_method1_save_dequantized.py` 스크립트를 통해 HF 8-bit 체크포인트를 float16으로 역양자화하여 저장하고, OCT 이미지 추론까지 성공한 전체 과정을 기록합니다.

### 4.1 문제 1: `pretrained_model_dir` 경로 오류

**증상:**
```
OSError: We couldn't connect to 'https://huggingface.co' to load the files,
and couldn't find them in the cached files.
```

**원인:**
`configs/paths/paths.yaml`의 `pretrained_model_dir`이 Windows 경로(`C:/Users/jjmin/SpecialistVLMs/saved_models`)로 설정되어 있어, Linux 환경에서 Llama-3 토크나이저를 HF 캐시에서 찾지 못함. `local_files_only=True`로 설정되어 있어 온라인 다운로드도 차단.

**해결:**
```yaml
# before
pretrained_model_dir: "C:/Users/jjmin/SpecialistVLMs/saved_models"

# after
pretrained_model_dir: "/home/ubuntu/bionexus/jgy/.cache/huggingface/hub"
```

### 4.2 문제 2: Dequantization 텐서 차원 불일치

**증상:**
```
RuntimeError: The size of tensor a (14336) must match the size of tensor b (4096)
at non-singleton dimension 1
```

**원인:**
8-bit → float16 역양자화 시 SCB(scale) 텐서의 `unsqueeze` 방향이 잘못됨.

- weight shape: `(out_features, in_features)` = 예: `(4096, 14336)`
- SCB shape: `(out_features,)` = `(4096,)`
- `scb.unsqueeze(0)` → `(1, 4096)` → `(4096, 14336)`과 브로드캐스트 **불가**
- `scb.unsqueeze(1)` → `(4096, 1)` → `(4096, 14336)`과 브로드캐스트 **가능**

**해결:**
```python
# before
dequantized = tensor.float() * scb.float().unsqueeze(0) / 127.0

# after
dequantized = tensor.float() * scb.float().unsqueeze(1) / 127.0
```

### 4.3 문제 3: `save_pretrained()` 직렬화 실패 (DictConfig + set + tied weights)

세 가지 에러가 연쇄적으로 발생:

| 에러 | 원인 |
|------|------|
| `TypeError: Object of type DictConfig is not JSON serializable` | Hydra `OmegaConf.DictConfig`가 모델 config에 포함 |
| `TypeError: Object of type set is not JSON serializable` | `_tied_weights_keys` 속성이 `set` 타입 |
| `RuntimeError: Some tensors share memory` | Visual Encoder에서 `model.layer3.*`와 `feature_tokens_model.6.*`가 동일 텐서를 공유 (tied weights). safetensors가 이를 거부 |

**해결:**
`save_pretrained()` 전체를 포기하고 `torch.save()`로 수동 저장하는 방식으로 전환. `torch.save`는 tied weights를 자연스럽게 처리.

```python
# 저장
state_dict = model.state_dict()
torch.save(state_dict, os.path.join(save_dir, "model.pt"))

# 로드 (from_pretrained 대신)
model = RetinaVLM(rvlm_config)
state_dict = torch.load(os.path.join(save_dir, "model.pt"), map_location="cpu")
model.load_state_dict(state_dict, strict=False)
```

### 4.4 문제 4: 추론 시 이미지 채널 불일치

**증상:**
```
RuntimeError: Given groups=1, weight of size [64, 1, 7, 7],
expected input[1, 192, 192, 192] to have 1 channels, but got 192 channels instead
```

**원인:**
OCT 이미지가 RGBA(4채널)로 로드됨. `convert_any_image_to_normalized_tensor`에서 RGBA→RGB(3채널) 변환은 하지만 1채널로는 변환하지 않음. Visual Encoder의 첫 Conv layer는 1채널(그레이스케일)을 기대.

**해결:**
```python
# before
image = np.array(Image.open(image_path))

# after
image = np.array(Image.open(image_path).convert('L'))
```

### 4.5 최종 동작 흐름

```
[Step 1] 최초 1회: Dequantized 체크포인트 생성
  HF 8-bit checkpoint 다운로드 (2 shards)
  → int8 텐서 224개를 float16으로 역양자화
  → torch.save()로 model.pt 저장 (1220 tensors)

[Step 2] 이후 매번: 빠른 로드
  RetinaVLM 모델 구조 생성
  → torch.load()로 state_dict 로드
  → load_state_dict() 적용 (Missing: 0, Unexpected: 0)

[Step 3] 추론
  OCT 이미지를 그레이스케일(L)로 로드 → 192×192 자동 리사이즈
  → model.forward([image], [query]) 호출
  → 임상 소견 텍스트 생성
```

### 4.6 추론 결과 예시

```
Input:  192×192 OCT grayscale image
Query:  "Write a detailed clinical report describing this OCT scan.
         Identify any visible biomarkers such as drusen, fluid, or atrophy."

Output: "The OCT scan reveals a large drusenoid PED with a hyporeflective core.
         There is moderate SHRM and moderate hypertransmission beneath the
         subretinal fluid. Additionally, subretinal fluid is present between
         the PED and the SHRM. No intraretinal fluid is observed.
         These findings suggest active late wet AMD."
```

### 4.7 수정된 파일 요약

| 파일 | 변경 내용 |
|------|-----------|
| `configs/paths/paths.yaml` | `pretrained_model_dir`을 Linux HF 캐시 경로로 변경 |
| `load_method1_save_dequantized.py` | unsqueeze 방향 수정, torch.save/load 방식 전환, 그레이스케일 변환 추가, 추론 코드 추가 |

---

## 5. Meta Init + Manual Load 방식 (방법 3) 문제 해결 과정 (v1.3)

`load_method3_meta_init_manual_load.py` 및 `retinavlm_meta_init_inference.ipynb`를 통해 검증된 방법.
기존 방법 1(`load_method1_save_dequantized.py`)과 달리, 중간 `model.pt` 저장 없이 HF 체크포인트를 직접 로드합니다.

### 5.1 방법 3의 핵심 아이디어

| 단계 | 기존 방식 (방법 1) | 방법 3 (Meta Init) |
|------|---------------------|---------------------|
| LLM 생성 | `from_pretrained` (base 가중치 로드) | `from_config` (빈 껍데기만 생성) |
| 체크포인트 적용 | base 위에 덮어쓰기 | 빈 모델에 직접 로드 |
| 중간 저장 | `torch.save()` → `model.pt` 필요 | 불필요 (HF 캐시에서 직접) |
| 메모리 피크 | base + checkpoint 이중 로드 | checkpoint만 로드 |

**장점:**
- Base Llama3 가중치를 로드하지 않으므로 **메모리 절약** (약 16GB)
- `save_pretrained()` 관련 직렬화 문제 완전 회피 (DictConfig, set, tied weights)
- 중간 `model.pt` 파일 불필요 → 디스크 절약

### 5.2 문제 1: Dequantization unsqueeze 방향 오류

**증상:**
```
RuntimeError: The size of tensor a (14336) must match the size of tensor b (4096)
at non-singleton dimension 1
```

**원인:**
방법 1에서 이미 수정한 `unsqueeze` 방향 문제가 방법 3 코드에도 동일하게 존재.
bitsandbytes int8 양자화의 SCB(Scale Column B)는 per-row scale이므로:

- weight shape: `(out_features, in_features)` = 예: `(4096, 14336)`
- SCB shape: `(out_features,)` = `(4096,)`
- `scb.unsqueeze(0)` → `(1, 4096)` → broadcast 불가
- `scb.unsqueeze(1)` → `(4096, 1)` → broadcast 가능

**해결:**
```python
# before
deq = tensor.float() * scb.float().unsqueeze(0) / 127.0

# after
deq = tensor.float() * scb.float().unsqueeze(1) / 127.0
```

### 5.3 문제 2: CUDA OOM (GPU 메모리 부족)

**증상:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 112.00 MiB.
GPU 0 has a total capacity of 79.19 GiB of which 73.94 MiB is free.
```

**원인:**
GPU 0에 다른 프로세스(VLLM 엔진 등)가 56.75 GiB를 점유 중이라 Llama3-8B 모델을 올릴 여유가 없음.

**해결:**
다른 GPU 프로세스 종료 후 재실행. 방법 3은 base 가중치 이중 로드를 하지 않으므로 기존 방식 대비 GPU 메모리 요구량이 낮음.

### 5.4 문제 3: Missing keys 556개 (llama_model, visual_encoder)

**증상:**
```
Missing (모델에는 있지만 체크포인트에 없음): 556
  Missing [llama_model]: 291
  Missing [visual_encoder]: 265
```

**원인:**
체크포인트의 키 이름 구조와 모델 내부 구조의 차이:
- 체크포인트: `model.llm.model.layers.0.*` (611개)
- 모델 내부 alias: `model.llama_model.layers.0.*` (추가 291개)
- Visual Encoder: 체크포인트에 별도로 포함되지 않음 (265개)

**영향:**
- LLM 부분: `llm.*` 경로로 이미 611개 키가 모두 로드되었으므로 `llama_model.*`은 alias(동일 참조)로 실질적 영향 없음
- Visual Encoder: `from_config` 시 랜덤 초기화되지만, 체크포인트 내 `model.visual_encoder.*` 키가 포함되어 있어 정상 로드됨 (all-zero params: 0/159)

**검증 결과:**
```
Visual Encoder: 0/159 all-zero params ← 정상
llama_proj.weight: mean=-0.000028, std=0.045305 ← 정상
llm.model.embed_tokens.weight: mean=0.000018, std=0.009307 ← 정상
```

### 5.5 추론 결과 검증

```
Input:  192x192 OCT grayscale image
Query:  "Write an extensive report describing the OCT image and listing
         any present biomarkers or other observations."

Output: "The OCT image reveals a large drusenoid PED with a hyporeflective
         core. There is moderate SHRM and a small amount of subretinal fluid
         between the PED and the SHRM. Additionally, there is a small amount
         of subretinal hyperreflective material and some intraretinal
         hyperreflective foci. These findings suggest active late wet AMD."
```

방법 1의 추론 결과와 동일한 수준의 임상 소견을 생성 → 방법 3 정상 동작 확인.

### 5.6 방법별 비교 요약

| 항목 | 방법 1 (save_dequantized) | 방법 3 (meta_init) |
|------|---------------------------|---------------------|
| 스크립트 | `load_method1_save_dequantized.py` | `load_method3_meta_init_manual_load.py` |
| 노트북 | `retinavlm_dequantized_inference.ipynb` | `retinavlm_meta_init_inference.ipynb` |
| LLM 초기화 | `from_pretrained` (base 로드) | `from_config` (빈 껍데기) |
| 중간 저장 | `model.pt` 필요 (약 16GB) | 불필요 |
| 최초 실행 시간 | 느림 (base 로드 + checkpoint) | 빠름 (checkpoint만) |
| 이후 실행 시간 | 빠름 (model.pt에서 로드) | 매번 역양자화 필요 |
| 추론 품질 | 정상 | 정상 (동일) |

### 5.7 수정/추가된 파일 요약

| 파일 | 변경 내용 |
|------|-----------|
| `load_method3_meta_init_manual_load.py` | unsqueeze 방향 수정 (`unsqueeze(0)` → `unsqueeze(1)`), 추론 코드 추가 |
| `retinavlm_meta_init_inference.ipynb` | 방법 3 기반 전체 추론 파이프라인 노트북 신규 생성 |

---

## 6. Override Load 방식 (방법 2) - 가장 난해했던 디버그 (v1.4)

`load_method2_override_load.py`를 통해 검증된 방법.
**이 파일만 수정 가능**한 제약 조건에서, HF 체크포인트를 직접 로드하여 추론까지 성공한 과정.

### 6.1 문제의 본질: 왜 디버그가 힘들었나

이 방법은 **5단계에 걸친 연쇄 문제**가 있었고, 각 단계를 해결해도 다음 문제가 드러나는 구조였다.
특히 마지막 문제(int8 역양자화 품질 저하)는 **가중치 통계가 정상으로 보여도 추론이 실패**하는, 가장 디버그하기 어려운 유형이었다.

### 6.2 문제 1: HF `from_pretrained` 내부 meta device 충돌

**증상:**
```
RuntimeError: You are using `from_pretrained` with a meta device context manager.
This is an anti-pattern as `from_pretrained` wants to load existing weights.
```

**원인:**
`_load_pretrained_model` 오버라이드 방식으로 `RetinaVLMWithDequant.from_pretrained()`을 호출하면,
HF 내부에서 meta device context를 설정. 이 안에서 Llama3의 `from_pretrained`이 중첩 호출되어 충돌.

**해결:**
`from_pretrained` 자체를 포기. 모델을 `MiniGPT4Module(config)`로 직접 생성하고, 체크포인트를 수동 로드.

### 6.3 문제 2: 3중 키 리매핑 (가장 복잡)

**증상:** 체크포인트 611개 키 vs 모델 1220개 키, 직접 매칭 0개

**원인:** MiniGPT4 내부의 다중 alias 구조:

```
체크포인트 키                              모델 키 (2개씩 존재)
─────────────────────────────────────────  ──────────────────────────────────
model.llm.model.model.layers.0.q_proj.w   llama_model.model.layers.0.q_proj.w
                                          llm.model.model.layers.0.q_proj.w

model.visual_encoder.model.conv1.weight   visual_encoder.model.conv1.weight
                                          visual_encoder.feature_tokens_model.0.weight
```

**필요했던 리매핑 4단계:**
1. `"model."` prefix 제거
2. `llm.model.model.` → `llama_model.model.` 변환
3. 양방향 alias 복제 (llama_model ↔ llm.model)
4. ResNet named children → Sequential 인덱스 (`conv1`→`0`, `bn1`→`1`, `layer1`→`4`, ...)

### 6.4 문제 3: `feature_tokens_model`이 별도 모듈

**증상:** 1220/1220 키 매칭 성공, Visual Encoder 0 all-zero → 정상처럼 보이나 추론 garbage

**원인:**
```python
# PretrainedResNet.__init__()
self.model = encoder.eval()
self.feature_tokens_model = nn.Sequential(*list(self.model.children())[:-1])
```

`feature_tokens_model`은 `model`의 children으로 구성된 **별도 nn.Sequential**.
`state_dict()`에서 별도 키로 등록됨. `model.*`로 로드해도 `feature_tokens_model.*`은 빈 상태.

**해결:** ResNet children 이름→인덱스 매핑 추가:
```python
resnet_name_to_idx = {
    "conv1": "0", "bn1": "1", "relu": "2", "maxpool": "3",
    "layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7",
}
# visual_encoder.model.conv1.weight → visual_encoder.feature_tokens_model.0.weight
```

### 6.5 문제 4: weight_format=0 오해 (오판)

**증상:** 역양자화 결과와 bitsandbytes 내장 함수 결과가 동일한데도 추론 garbage

**오판:** `weight_format=0`이 col_turing GPU 레이아웃이라 undo 필요하다고 추정

**실제:**
```python
LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING = {
    "row": 0,        # ← weight_format=0 은 표준 row-major!
    "col32": 1,
    "col_turing": 2,
    "col_ampere": 3
}
```
→ 레이아웃 변환 불필요, 역양자화 자체는 정확했음

### 6.6 문제 5: Int8 역양자화 LLM 가중치의 품질 저하 (핵심)

**이것이 가장 디버그하기 어려웠던 이유:**
- 모든 키 매칭 완료 (1220/1220)
- 가중치 통계 정상 (mean≈0, std≈0.02)
- 역양자화 수학적으로 정확 (bitsandbytes 결과와 diff=0)
- **그런데 추론 결과가 garbage**

**핵심 발견 - 전/후 비교 실험:**
```
[체크포인트 로드 전] "The capital of France is"
→ "Paris, which is located in the north-central part of the country."  ✅

[체크포인트 로드 후] "The capital of France is"
→ "the most beautiful city in the world, and the city is the most beautiful..."  ❌

q_proj weight diff (before vs after): max=0.718, mean=0.006
```

**원인:**
- RetinaVLM 학습 시 **LLM은 frozen** (가중치 변경 없음)
- 체크포인트의 LLM 가중치 = 원본 Llama3의 int8 양자화 → 역양자화 버전
- 개별 가중치 오차는 작지만, 32개 transformer 레이어를 거치며 **오차 증폭**
- 역양자화 fp16 << 원본 fp16 품질

### 6.7 최종 해결 전략

```
Step 1: Llama3-8B-Instruct 사전학습 가중치로 모델 생성 (initialize=True)
        → 원본 fp16 LLM 가중치 확보

Step 2: HF 체크포인트 다운로드 + int8→fp16 역양자화

Step 3: LLM 가중치(582개)는 제외하고, visual_encoder + llama_proj(638개)만 로드
        → 원본 LLM 가중치 유지
```

```python
# LLM frozen이므로 체크포인트의 역양자화 LLM 가중치 대신 원본 유지
llm_prefixes = ("llm.", "llama_model.")
filtered_dict = {k: v for k, v in remapped_dict.items()
                 if not any(k.startswith(p) for p in llm_prefixes)}
```

### 6.8 최종 결과

| 항목 | 값 |
|------|-----|
| 키 리매핑 후 매칭 | 1220/1220 |
| 실제 로드 (LLM 제외) | 638 (visual_encoder + llama_proj) |
| Visual Encoder all-zero | 0/159 |
| llama_proj 체크포인트 일치 | max diff = 0.000000 (완벽 일치) |
| LLM 단독 테스트 | "Paris, which is located in the north-central..." ✅ |
| OCT 추론 | "The retinal OCT image shows no significant abnormalities or biomarkers." ✅ |

### 6.9 교훈

1. **가중치 통계(mean/std)가 정상이어도 추론이 실패할 수 있다** — 개별 오차가 작아도 deep network에서 누적됨
2. **Frozen 레이어의 역양자화는 불필요하고 유해하다** — 원본 fp16 >> 역양자화 fp16
3. **전/후 비교 테스트가 가장 효과적인 디버그** — 체크포인트 로드 전후 동일 입력으로 출력 비교
4. **nn.Module alias와 nn.Sequential 래핑은 state_dict 키를 분리** — 같은 텐서를 공유해도 별도 키로 등록됨
5. **weight_format 숫자의 의미를 소스코드에서 직접 확인** — 추측하지 말 것

---
**보고서 업데이트일**: 2026-03-10
**v1.4 작성자**: Claude Code
