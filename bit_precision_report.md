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
**보고서 작성일**: 2026-03-08  
**작성자**: Gemini CLI Assistant
