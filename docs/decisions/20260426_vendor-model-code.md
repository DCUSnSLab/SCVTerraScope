# ADR: SCVTerraVision 모델 코드를 SCVTerraScope 에 vendor

- **상태**: 확정 — `20260426_consume-terravision-editable.md` 를 **대체** (Supersedes)
- **결정일**: 2026-04-26
- **관련 단계**: Phase 1-1 GUI 구현 (이번 세션 중간에 방향 전환)

## 배경

Phase 1-1 셋업 과정에서 `pip install -e ../SCVTerraVision` 방식으로는 이상한 우회가 누적되는 것이 드러났다.

| 발견된 마찰 | 원인 |
|---|---|
| `pip install -e ../SCVTerraVision` 후에도 `from models.* import` 실패 | SCVTerraVision `pyproject.toml` 의 `[tool.setuptools] packages = []`. editable install 이 메타데이터만 등록하고 실제 모듈 경로는 노출 안 함. |
| `_terravision_path.py` sys.path injection helper 필요 | 위 문제 우회. 임시방편. |
| `transformers>=4.56,<5.0` + `timm>=1.0` 핀이 SCVTerraScope 에 강제 | SCVTerraVision 측 의존이지만 SCVTerraVision 의 `transformers>=4.56` 만 있어 SCVTerraScope 가 따로 핀 필요. |
| SCVTerraVision 측 브랜치 (objectdetection vs ddp-training vs main) 가 어떤지에 따라 SCVTerraScope 가 동작/실패 | Tight coupling. |
| GPU 모니터링 도구가 학습 저장소 의존이라는 사실 자체가 어색 | 책임 분리 위반. |

본 ADR 은 위 마찰을 모두 제거하기 위해 **inference 에 필요한 SCVTerraVision 코드 일부를 SCVTerraScope 안으로 vendor** 한다.

## 결정

다음 자산을 SCVTerraScope 저장소로 복사한다.

| SCVTerraVision 원본 | SCVTerraScope 대상 | 비고 |
|---|---|---|
| `models/backbone/dinov3_backbone.py` | `src/scvterrascope/model/backbone.py` | 거의 verbatim. import 무관. |
| `models/detection/detr_head.py` | `src/scvterrascope/model/detr_head.py` | 내부 import 만 `scvterrascope.model.backbone` 으로 갱신. |
| `training/train_detection.py` 의 `letterbox_resize` + `IMAGENET_MEAN/STD` | `src/scvterrascope/inference/preprocess.py` | inline 함수 + 모듈 상수. ~30 줄. |
| `configs/dataset/coda_taxonomy.yaml` | `src/scvterrascope/data/coda_taxonomy.yaml` | `operational_classes` + `version` 만 유지. CODa→operational/BDD100K 변환 표는 학습-only 라 제거. |

총 vendor 규모: ~570 줄 + 1 YAML.

**제거되는 것**:
- `pip install -e ../SCVTerraVision` 절차
- `src/scvterrascope/_terravision_path.py` (sys.path 주입 helper)
- `pyproject.toml` 의 SCVTerraVision 관련 코멘트
- `AppConfig.terravision_root`, `InferenceEngine.__init__(..., terravision_root=...)` 인자
- `TERRAVISION_ROOT` 환경변수 의존
- runbook 의 "SCVTerraVision 브랜치 체크아웃" 단계

## 트레이드오프

**얻는 것**:
1. **독립 설치** — `pip install -e .` 한 줄로 SCVTerraScope 자체로 동작.
2. **명확한 책임 분리** — SCVTerraScope = 추론/시각화, SCVTerraVision = 학습/평가. 서로 build 시점에 서로의 환경을 강요하지 않음.
3. **테스트 깨끗** — 19 passed, 0 skipped (이전엔 SCVTerraVision 가용성에 따른 conditional skip 2개).
4. **PyPI 빌드 가능성** — 다른 머신에서 wheel 로 설치/배포 가능.
5. **관심사 분리** — 추론 도구가 학습 저장소의 progress 문서나 학습 dataset 코드까지 끌어오지 않음.

**잃는 것**:
1. **수동 sync 책임** — SCVTerraVision 이 모델 아키텍처를 변경하면 (예: Phase 1-2c 에서 multi-scale feature 추가) vendored 코드를 수동으로 갱신해야 한다. 학습된 체크포인트와 호환되는 architecture 가 어긋나면 `model.load_state_dict()` 가 missing/unexpected key 폭주를 낸다.

## Sync 정책

SCVTerraVision 가 다음 중 하나라도 변경하면 본 vendored 파일들도 동시에 갱신해야 한다:

1. `models/backbone/dinov3_backbone.py` — DINOv3 백본 출력 형태/구조
2. `models/detection/detr_head.py` — DeformableDetr config 또는 shim 구조
3. `training/train_detection.py` 의 `letterbox_resize` / `IMAGENET_MEAN` / `IMAGENET_STD`
4. `configs/dataset/coda_taxonomy.yaml` 의 `operational_classes` 클래스 추가/제거/이름 변경

각 vendored 파일 헤더에 "Vendored from … @ 2026-04-26" 명시. sync 작업 시:

```bash
# 차이 빠르게 비교
diff -u ~/development/SCVTerraVision/models/backbone/dinov3_backbone.py \
        ~/development/SCVTerraScope/src/scvterrascope/model/backbone.py

diff -u ~/development/SCVTerraVision/models/detection/detr_head.py \
        ~/development/SCVTerraScope/src/scvterrascope/model/detr_head.py
```

차이가 의도된 변경이면 vendor 파일을 갱신하고 헤더의 날짜를 갱신, ADR 에 한 줄 sync 노트 append. 차이가 의도되지 않은 drift 면 SCVTerraVision 측 수정 또는 SCVTerraScope 측 수정 중 어느 쪽이 옳은지 결정.

## 영향 / 후속 작업

- **삭제됨**: `_terravision_path.py`, runbook 의 "0) SCVTerraVision 브랜치 체크아웃" 절, AppConfig 의 `terravision_root` 필드.
- **갱신됨**: pyproject 의존성 목록, runbook 의 설치 절차, phase1-1 progress 문서의 환경 메모.
- 본 ADR 은 `20260426_consume-terravision-editable.md` 를 supersede. 그 ADR 은 그대로 두고 헤더만 "Superseded by 20260426_vendor-model-code" 로 갱신.
- **CI/CD 도입 시점**: SCVTerraScope 만 체크아웃하면 충분하다. SCVTerraVision 은 별도 워크플로우로 관리.
- **장기**: SCVTerraVision 측에서 inference 모듈을 별도 PyPI 패키지로 분리하면 본 ADR 을 다시 한 번 supersede 해서 import 기반으로 회귀할 수 있다. 그 결정 시점은 SCVTerraVision 가 stable architecture 에 도달했을 때.
