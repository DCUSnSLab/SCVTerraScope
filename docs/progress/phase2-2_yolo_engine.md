# Phase 2-2 — YOLO11/12 엔진 추가 + 모델 선택기

- **상태**: ✅ 완료·검토대기 (2026-04-27)
- **시작일**: 2026-04-27
- **완료일**: 2026-04-27 (사용자 X11 GUI 검증 대기)
- **관련 ADR**: 없음 (Ultralytics 채택은 ADR 미필요)

## 목표

기존 DINOv3+DETR 엔진 옆에 Ultralytics YOLO (11/12 시리즈) 를 추가하고, 동일 GUI 안에서 dropdown 으로 전환할 수 있게 한다. SCV/CODa 데이터에 두 모델 결과를 직접 비교하기 위한 도구.

## 결정 (2026-04-27)

| 항목 | 선택 |
|---|---|
| YOLO 버전 | **YOLO12** 우선, YOLO11 도 함께 선택 가능 |
| 가중치 소스 | Ultralytics COCO 80-class 사전학습 (자동 다운로드) |
| 모델 선택 UI | Controls 패널 단일 dropdown (사이드 비교 X) |
| 용도 | DINOv3+DETR (CODa 16-class) vs YOLO (COCO 80-class) 성능/정확도 비교 |

## 아키텍처

```
inference/
├─ engine.py              ← BaseInferenceEngine Protocol 추가 + 기존 InferenceEngine (DETR)
├─ yolo_engine.py         ← 신규: Ultralytics 래퍼
├─ factory.py             ← 신규: model_kind → 엔진 인스턴스 (MODEL_REGISTRY)
└─ __init__.py            ← BaseInferenceEngine, YoloEngine, build_engine, model_choices export
gui/
├─ config.py              ← AppConfig.model_kind ("dinov3_detr" 기본)
├─ widgets/control_panel.py  ← Model dropdown 추가, set_class_names() 동적 재구성
└─ main_window.py         ← build_engine 사용, 모델 변경 시 taxonomy/palette/class-grid 재빌드
```

`Detection`, `InferenceResult` 는 기존 그대로 — 두 엔진이 동일 dataclass 반환해 GUI / Performance / draw_detections 가 변경 없이 동작.

## 핵심 디테일

### BaseInferenceEngine Protocol

`@runtime_checkable Protocol` 로 GUI 가 의존하는 surface 를 명시. 두 엔진이 구조적으로 만족:
- 메서드: `is_loaded`, `load`, `predict`
- 속성: `checkpoint_path`, `image_size`, `top_k`, `aspect_crop_mode`, `pad_position`
- 프로퍼티: `device`, `device_name`, `epoch`, `taxonomy`, `param_count`, `gpu_total_mb`

### YoloEngine 의 Letterbox

Ultralytics 가 내부에서 letterbox 처리. `aspect_crop_mode` / `pad_position` 인자는 받아두지만 무시 (DETR 와의 인터페이스 parity 위해). `imgsz` 만 전달. Ultralytics Results 의 `.speed` dict 에서 preprocess/inference/postprocess ms 를 분리해 PerformancePanel 로 전달.

### Class Set 차이 처리

- DETR: CODa 16 클래스 (vendored YAML 로 Taxonomy 빌드)
- YOLO: COCO 80 클래스 (`model.names` dict 에서 Taxonomy 빌드)
- ControlPanel 의 class checkbox 그리드는 모델 변경 시 `set_class_names()` 로 재빌드. 80개라 ScrollArea 안에 2-column grid 로 wrap.
- `class_id` 는 둘 다 1-indexed (CODa COCO category id 와 일치). 색상 팔레트는 `palette_for(max(16, len(taxonomy)))` 로 자동 확장.

### 모델 선택 UX

- ControlPanel 의 Engine 박스에 "Model:" combo 추가 — 엔진 패밀리 + 라벨 표시.
- 변경 시 `_on_model_changed(model_kind)` 가 엔진 invalidation. 다음 추론 트리거 (Run Inference / 파일 선택 / bag step) 에서 `_ensure_engine` 가 새 모델로 reload.
- DINOv3+DETR 은 `checkpoint_path` 필수, YOLO 는 자동 다운로드라 빈 칸 OK.

## 단위 테스트

- `tests/test_factory.py` (6 passed):
  - MODEL_REGISTRY 가 dinov3 + yolo12 + yolo11 키 보유
  - `family_for` 분기
  - `build_engine("yolo12n", checkpoint_path=None)` → checkpoint_path = "yolo12n.pt" 자동 사용
  - `build_engine("dinov3_detr", checkpoint_path=None)` → ValueError
  - 두 엔진이 GUI 가 의존하는 메서드/속성 surface 동일
- 전체: **51 passed** (45 + 6 신규)

## 비교 결과 — 동일 입력 양쪽 엔진

```
=== CODa 1224×1024 (캠퍼스 보행) ===
  DINOv3+DETR  fwd=410ms  total=492ms  fps= 2.0  >=0.3: 5
    bollard 0.929, bollard 0.878, sign 0.873, bollard 0.868, barrier 0.862
  YOLO12n      fwd= 42ms  total= 64ms  fps=15.7  >=0.3: 0
                                                  (COCO에 bollard/sign post 없음)

=== SCV 1920×1080 (차량 대시보드) ===
  DINOv3+DETR  fwd=214ms  total=239ms  fps= 4.2  >=0.3: 3
    vehicle 0.926, vehicle 0.834, vehicle 0.370
  YOLO12n      fwd= 39ms  total= 52ms  fps=19.4  >=0.3: 5
    car 0.905, car 0.895, car 0.891, car 0.667, car 0.454
```

명확한 패턴:
- **YOLO12n 이 5-10× 빠름** (nano 사이즈 + Ultralytics 의 효율적 letterbox+NMS)
- **CODa 캠퍼스 인프라**: COCO 사전학습은 bollard/barrier/sign post 같은 클래스를 모르니 검출 불가 — DINOv3+DETR 의 도메인 학습 우위
- **SCV 차량 대시보드**: COCO 의 'car/truck/bus' 가 일반 차량 분포에 학습되어 YOLO 가 더 많이 잡고 더 빠름

오버레이 산출물: `outputs/compare_{coda,scv}_{dinov3_detr,yolo12n}.png` (4개).

## 사용자 X11 GUI 검증 시나리오

1. Controls → Engine → **Model: dropdown** 에서 "YOLO12 nano" 선택.
2. 첫 실행 시 Ultralytics 가 yolo12n.pt 자동 다운로드 (5MB).
3. ROS Bag 탭에서 ⏯ Play → autoplay (drop-frame 적용된 ~15 FPS 기대).
4. Class 체크박스 80개로 자동 재구성됨 — person/car/truck/.../traffic light.
5. PerformancePanel 의 "Model / device" 영역에서 model 명/param count 변경 확인.
6. "DINOv3+DETR" 로 다시 전환 — class 16개로 재구성, Detection 결과 다름 확인.

## 알려진 한계 / 후속

- **사이드-바이-사이드 비교 뷰** (좌우 캔버스에 두 모델 동시 표시) 는 현 단계 범위 밖. 한 번에 한 모델만 활성. 사용자가 보고싶다면 별도 단계 (Phase 2-3).
- **YOLO 커스텀 .pt 학습**: SCV 데이터로 fine-tune 한 .pt 가 생기면 `Browse` 로 직접 가리키면 됨 — `family_for("yolo*")` 가 yolo 로 분기되어 동작. 단 클래스 셋이 다르면 `model.names` 가 변경된 dict 라 자동으로 적응.
- **메모리**: 두 모델 동시 GPU 적재는 X. 변경 시 이전 엔진 dispose + 새 엔진 load. RTX 4060 Ti 8GB 에 둘 다 올리려면 사이드 비교 뷰 만들 때 같이 고민.

## 사용자 검토 결과

(사용자 작성 영역)
