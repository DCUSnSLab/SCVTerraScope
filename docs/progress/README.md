# 진행 상황 대시보드

본 문서는 SCVTerraScope 개발의 단계별 상태를 한눈에 보기 위한 인덱스이다. 세부 체크리스트·검증 로그·사용자 검토 결과는 각 `phase*.md` 에서 관리한다.

## 상태 범례

- ⏳ 진행중
- ✅ 완료·검토대기 (사용자 검토 대기 상태)
- 🟢 승인완료 (머지 및 다음 단계 진입 가능)
- ⛔ 블록
- ⚪ 예정 (아직 시작 전)

## 대시보드

| # | 단계 | 상태 | 파일 | 비고 |
|---|------|-----|------|------|
| 0 | 저장소 골격 + 문서 구조 | ✅ 완료·검토대기 (2026-04-26) | [phase0.md](phase0.md) | pyproject + docs 골격 + ADR (PyQt6, editable install) |
| 1-0 | 검증 데이터 샘플 확보 (CODa tiny) | ✅ 완료·검토대기 (2026-04-26) | [phase1-0_data_samples.md](phase1-0_data_samples.md) | 9.1GB tiny split 다운로드 + 압축 해제 완료. 22 시퀀스, 1620 cam0 PNG 프레임. 실 CODa 프레임 추론 검증 완료. |
| 1-1 | 이미지 GUI 모니터 | ✅ 완료·검토대기 (2026-04-26) | [phase1-1_image_monitor.md](phase1-1_image_monitor.md) | PyQt6 GUI + vendored DINOv3+DETR + 19 pytest green. **HF_TOKEN 불필요** — backbone 을 from_config 로 빌드. 실 CODa 프레임 추론 OK (bollard/sign/barrier 0.86–0.93). |
| 2 | ROS 2 스트림 모니터 | ⚪ 예정 | — | Phase 1-1 승인 후 별도 재계획 |

## 갱신 규칙

1. 단계가 `진행중 → 완료·검토대기` 로 바뀌면 해당 행의 상태와 `파일` 링크를 즉시 갱신한다.
2. 사용자 검토에서 승인되면 상태를 🟢 로 바꾸고, 다음 단계 행의 상태를 ⏳ 로 전환 + 새 `phase*.md` 를 생성한다.
3. 중요 기술 결정은 `docs/decisions/` 에 ADR 을 추가하고, 여기 `비고` 칼럼에 파일명 1줄 참조만 남긴다.
