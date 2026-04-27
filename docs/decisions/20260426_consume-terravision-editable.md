# ADR: SCVTerraVision 을 editable install 로 소비

- **상태**: ⛔ Superseded by [`20260426_vendor-model-code.md`](20260426_vendor-model-code.md) (당일 후반 결정 변경)
- **결정일**: 2026-04-26 (오전)
- **관련 단계**: Phase 1 (이미지 GUI), 이후 모든 Phase

> **참고**: 본 ADR 은 Phase 1-1 셋업 중 누적된 마찰 (sys.path 우회, packages=[] 문제, 환경변수 의존) 으로 인해 같은 날 후반에 vendor 방식으로 변경되었다. 결정 변경 사유는 supersede ADR 참고.

---

## 배경

SCVTerraScope 는 추론 시 SCVTerraVision 의 다음 자산을 필요로 한다.

- `models/detection/detr_head.py` — `DinoV3DeformableDetr` 모델 빌더 + checkpoint state_dict 호환
- `models/backbone/dinov3_backbone.py` — DINOv3 백본 래퍼
- `training/train_detection.py` — letterbox preprocess (좌표 역변환과 1:1 일치 필요)
- `configs/dataset/coda_taxonomy.yaml` — 16-class 라벨 정의

이 자산들은 SCVTerraVision 의 학습 진행에 따라 계속 진화한다 (Phase 1-2c, 1-3, 1-4 …). SCVTerraScope 가 어떻게 이를 소비할지 결정해야 한다.

## 검토한 대안

| 대안 | 장점 | 단점 |
|---|---|---|
| **`pip install -e ../SCVTerraVision` (editable)** | 학습측 변경 자동 반영, 코드 중복 0, `import models.detection.detr_head` 가 그대로 동작 | sibling 디렉토리 가정 (개발자 환경 종속), CI 에서 별도 체크아웃 필요 |
| 코드 사본 (vendoring) | SCVTerraScope 단독 빌드 가능, 외부 저장소 무관 | 학습측 업데이트마다 수동 동기화, drift 발생 시 좌표/라벨 불일치 위험 |
| ONNX/TorchScript export 후 소비 | 런타임 의존성 최소, 배포 간단 | export 파이프라인 선행 구축 필요, 학습 모델 변경 시 매번 export |

## 결정

**`pip install -e ../SCVTerraVision` 을 채택한다.** 이유:

1. **단일 진실 원천** — letterbox 규칙 · taxonomy · 모델 정의가 SCVTerraVision 한 곳에서만 관리되어 좌표 역변환 버그 / 라벨 mismatch 위험이 사라진다.
2. **학습 진행 자동 추적** — Phase 1-2c (캠퍼스 파인튠) 등에서 모델 정의가 바뀌어도 SCVTerraScope 측 코드 수정이 거의 없다.
3. **개발자 환경 일관성** — `marsberry@cu.ac.kr` 의 작업 환경에서 두 저장소가 sibling 으로 체크아웃되는 패턴이 이미 형성되어 있다 (`~/development/SCVTerraVision`, `~/development/SCVTerraScope`).

ONNX export 는 후속 옵션으로 남긴다 — Phase 2 의 영상 스트림에서 PyTorch forward 가 FPS 병목으로 드러나면 별도 ADR 로 export 분기를 결정한다.

## 영향 / 후속 작업

- `README.md` 와 `docs/runbooks/phase1_launch.md` 에 다음 설치 순서 명시:
  ```bash
  pip install -e ../SCVTerraVision     # 모델 코드
  pip install -e .[dev]                # SCVTerraScope
  ```
- `pyproject.toml` 에는 SCVTerraVision 을 의존성으로 **명시하지 않는다** (경로 의존이므로 PyPI 빌드/배포 시 깨짐).
- SCVTerraVision 측에 호환성을 깨는 변경 (예: `DinoV3DeformableDetr` 시그니처 변경) 이 들어오면, SCVTerraScope 의 `phase*.md` 검증 로그에 영향 범위를 기록한다.
- CI/CD 도입 시점에서 SCVTerraVision 도 동일 워크플로우에서 체크아웃 + editable install 하는 단계 추가가 필요함을 본 ADR 의 후속 항목으로 남긴다.
