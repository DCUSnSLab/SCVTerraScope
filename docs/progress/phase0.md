# Phase 0 — 저장소 골격 및 문서 구조

- **상태**: ✅ 완료·검토대기 (2026-04-26)
- **시작일**: 2026-04-26
- **완료일**: 2026-04-26 (사용자 검토 대기)
- **담당 PR**: (작성 후 링크)
- **관련 ADR**: `docs/decisions/20260426_gui-framework-pyqt6.md`, `docs/decisions/20260426_consume-terravision-editable.md`

## 목표

본격 코드 작성 전, SCVTerraVision 의 문서·빌드 컨벤션을 미러링한 저장소 골격을 깔아 후속 Phase 의 작업 흐름을 표준화한다.

- src-layout 패키지 구조 + `pyproject.toml`
- 빌드/테스트 인프라 (`.gitignore`, `pytest.ini`, `.pre-commit-config.yaml`)
- 문서 트리 (`docs/PLAN.md`, `docs/progress/`, `docs/decisions/`, `docs/runbooks/`)
- 두 건의 핵심 ADR (GUI 프레임워크, SCVTerraVision 의존 방식)

## 확정된 결정 (2026-04-26)

1. **GUI 프레임워크**: **PyQt6** — ADR `20260426_gui-framework-pyqt6.md`.
2. **모델 의존성**: SCVTerraVision 을 sibling editable install 로 소비 — ADR `20260426_consume-terravision-editable.md`.
3. **ROS 버전 (Phase 2)**: **ROS 2** (Humble/Jazzy 가정). Phase 2 진입 시점에 배포판 확정.
4. **Phase 1 입력 범위**: 단일 이미지 + 폴더 일괄. 비디오/스트림은 Phase 2.

## 체크리스트

- [x] `pyproject.toml` (Python ≥3.10, PyQt6/pillow/numpy/opencv-python/pyyaml, ruff/black 설정)
- [x] `.gitignore` (Python · Qt · venv · outputs · 모델 weight · `.env`)
- [x] `.pre-commit-config.yaml` (ruff + ruff-format + black + pre-commit-hooks)
- [x] `pytest.ini`
- [x] `README.md` (설치 절차 + Phase 로드맵 링크)
- [x] `src/scvterrascope/{inference,visualization,gui,gui/widgets}/__init__.py` 빈 파일
- [x] `scripts/`, `configs/`, `tests/`, `outputs/` 디렉토리 (`.gitkeep`)
- [x] `configs/default.yaml` (체크포인트 placeholder + 임계값 + image_size)
- [x] `docs/PLAN.md` (`/home/soobin/.claude/plans/objectdetection-cached-hippo.md` 사본)
- [x] `docs/progress/README.md` (대시보드)
- [x] `docs/progress/phase0.md` (본 파일)
- [x] `docs/decisions/20260426_gui-framework-pyqt6.md`
- [x] `docs/decisions/20260426_consume-terravision-editable.md`
- [x] **(후속)** `docs/runbooks/data_setup.md` — Phase 1-0 에서 작성됨 (CODa tiny 다운로드 절차).
- [ ] **(후속, Phase 1-1 진입 시)** `docs/runbooks/phase1-1_launch.md` — HF_TOKEN, 체크포인트 경로, venv 셋업 순서.

## 산출물

- `pyproject.toml`, `.gitignore`, `.pre-commit-config.yaml`, `pytest.ini`, `README.md`
- `src/scvterrascope/` 패키지 골격 (코드 미작성)
- `configs/default.yaml`
- `docs/PLAN.md`, `docs/progress/{README.md, phase0.md}`, `docs/decisions/20260426_*.md`

## 검증 로그

- `python3 --version` → `Python 3.10.12` (요구사항 ≥3.10 충족, SCVTerraVision 과 일치).
- 시스템 검사 중 `/opt/ros/humble` 가 사이트 패키지로 노출됨을 확인 — Phase 2 의 ROS 2 Humble 타겟 가정이 맞음 (`rclpy` 가 별도 설치 없이 이용 가능). 본 사실은 Phase 2 진입 시 `phase2_ros2_stream.md` 에서 다시 확인한다.
- `python3 -c "import yaml; yaml.safe_load(open('configs/default.yaml').read())"` → dict 형태로 정상 로드. ✅
- `pyproject.toml` 은 시스템 Python (3.10.12) 에 `tomllib`/`tomli` 부재로 자동 파서 검증 불가. 구조는 SCVTerraVision `pyproject.toml` 패턴 그대로 미러. 실 검증은 Phase 1 진입 시 `pip install -e .[dev]` 의 wheel 빌드 단계에서 수행.
- 트리 점검: `find . -type f -not -path './.git/*' | sort` 로 위 체크리스트 파일 21개가 모두 존재함을 확인 (LICENSE 포함).
- 본 단계는 코드를 작성하지 않으므로 pytest 는 빈 `tests/` 에서 수집 0건으로 종료된다.

## 사용자 검토 결과

(사용자 작성 영역)
