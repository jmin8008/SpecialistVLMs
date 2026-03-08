# RetinaVLM-Specialist 모델 로딩 오류 해결 보고서

본 보고서는 `retfound_projection_llm_experiment.ipynb` 실행 중 발생한 주요 오류들과 그 해결 과정을 기록합니다.

---

## 1. 설정 참조 오류 (InterpolationKeyError)

### [현상]
노트북 실행 시 `Interpolation key 'paths.pretrained_model_dir' not found` 오류 발생.

### [원인]
1. **Hydra 메커니즘 미적용**: `OmegaConf.load`는 단일 파일만 읽어오며, `default.yaml`에 정의된 `defaults` 섹션(다른 설정 파일 병합)을 처리하지 못함.
2. **로컬 설정 파일 부재**: 실제 경로를 정의하는 `configs/paths/paths.yaml` 파일이 존재하지 않음.

### [해결 방법]
1. **파일 생성**: `template.yaml`을 복사하여 `paths.yaml`을 생성하고 프로젝트 루트 경로를 반영함.
2. **코드 수정**: 노트북의 설정 로드 방식을 `OmegaConf.load`에서 Hydra의 `initialize` 및 `compose` 방식으로 변경하여 모든 설정 참조(`${paths...}`)가 정상적으로 해결되도록 함.

---

## 2. 메타 장치 충돌 오류 (RuntimeError)

### [현상]
Llama3 모델 로딩 중 `RuntimeError: You are using from_pretrained with a meta device context manager...` 발생.

### [원인]
1. **중첩된 로딩 구조**: `RetinaVLM` -> `MiniGPT4` -> `Llama3` 순으로 모델을 불러오는 과정에서 `transformers` 라이브러리가 내부적으로 "설계도(Meta Device)" 모드를 활성화함.
2. **데이터 충돌**: 가짜 장치(Meta Device) 상태인 '설계도' 위에 실제 가중치(실제 벽돌)를 쌓으려고 시도하면서 발생한 충돌.

### [해결 방법]
- **장치 강제 초기화**: `models/llama3.py`에서 Llama3 모델을 불러오기 직전에 `torch.set_default_device('cpu')`를 호출함.
- **효과**: 가짜 설계도(Meta) 상태를 해제하고, 실제 데이터를 담을 수 있는 공간(CPU RAM)을 확보한 뒤 가중치를 안전하게 로드함.

---

## 3. 장치 할당 및 로딩 방식 설명

### [질문] 왜 CPU에서 로드한 뒤 GPU로 옮기나요?
- **범용성**: GPU가 없는 환경에서도 코드가 돌아갈 수 있도록 설계된 프로젝트의 유연성 때문임.
- **안전성**: Llama3와 같은 대형 모델은 GPU 메모리에 바로 올리다가 부족할 경우 로딩 자체가 실패할 수 있음. 따라서 용량이 큰 CPU(RAM)에 먼저 안전하게 올린 뒤 GPU로 이동하는 방식을 채택함.

---

## 4. 최종 상태
- 모든 설정 참조 정상 작동 확인.
- 모델 가중치 로딩 충돌 해결.
- 최종적으로 `model.to(DEVICE)`를 통해 GPU(CUDA) 활용 가능 상태 확인.

**작성일**: 2026년 3월 8일
**작성자**: Gemini CLI (SpecialistVLM Assistant)
