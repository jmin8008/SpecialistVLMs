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

## 2. 메타 장치 충돌 및 중첩 로딩 오류 (RuntimeError & NotImplementedError)

### [현상]
Llama3 모델 로딩 중 `RuntimeError` 발생 후, 수정을 거쳐도 `NotImplementedError: Cannot copy out of meta tensor; no data!` 발생.

### [원인]
1. **중첩된 로딩 구조**: 상위 모델인 `RetinaVLM.from_pretrained`가 실행될 때 `transformers` 라이브러리가 전체 모델 구조 파악을 위해 내부적으로 "가짜 장치(Meta Device)" 컨텍스트를 활성화함.
2. **컨텍스트 전이**: 이 가짜 장치 상태가 자식 모델인 `Llama3`에게도 강제로 적용되어, 진짜 가중치 데이터를 복사하려고 할 때 "가짜 공간에는 실제 데이터를 담을 수 없다"며 충돌 발생.
3. **권한 오류 (WinError 32)**: 다운로드 도중 프로세스 충돌로 인해 파일이 잠겨 `PermissionError` 발생.

### [최종 해결 방법]
1. **부모 모델의 로딩 방식 변경**: `RetinaVLM.from_pretrained`를 사용하는 대신, 모델 인스턴스를 직접 생성하고 가중치(`pytorch_model.bin`)를 수동으로 로드하도록 `models/retinavlm_wrapper.py` 수정.
2. **컨텍스트 차단**: 이 방식을 통해 `transformers`가 자동으로 생성하는 `meta` 장치 컨텍스트의 생성을 원천 봉쇄함.
3. **찌꺼기 파일 제거**: 다운로드 실패 시 생성된 `.incomplete` 및 `.lock` 파일을 수동으로 삭제하여 파일 접근 권한 문제 해결.

---

## 3. 장치 할당 및 로딩 방식 설명

### [질문] 왜 CPU에서 로드한 뒤 GPU로 옮기나요?
- **범용성**: GPU가 없는 환경에서도 코드가 돌아갈 수 있도록 설계된 프로젝트의 유연성 때문임.
- **안전성**: Llama3와 같은 대형 모델은 GPU 메모리에 바로 올리다가 부족할 경우 로딩 자체가 실패할 수 있음. 따라서 용량이 큰 CPU(RAM)에 먼저 안전하게 올린 뒤 GPU로 이동하는 방식을 채택함.

---

## 4. 최종 상태
- 모든 설정 참조 및 로컬 경로 정상 작동 확인.
- `meta` 장치 충돌 문제를 `from_pretrained` 우회 로직으로 완벽히 해결.
- 수동 가중치 로드 방식을 통해 안정적인 모델 합체 완료.
- 최종적으로 `model.to(DEVICE)`를 통해 GPU(CUDA) 활용 가능 상태 확인.

**작성일**: 2026년 3월 8일
**작성자**: Gemini CLI (SpecialistVLM Assistant)
