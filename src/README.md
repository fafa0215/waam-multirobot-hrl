# 코드 스냅샷 안내

이 폴더는 과제 공동 파이프라인 안에서 작성한 HRL 모듈을 그대로 옮긴 스냅샷입니다. 원래 위치의 import 경로(`pipeline.S3_path_planning.*`)를 바꾸지 않았기 때문에, 이 저장소만으로는 실행되지 않습니다.

## 이 폴더의 모듈 (직접 작성)

| 파일 | 역할 |
|---|---|
| `hrl/S36_legacy_xy_collision.py` | 원본 HRL의 XY 박스·선분 충돌식을 새 형상 경로에 적용 |
| `hrl/S36_hrl_layerwise.py` | 초기 층 단위 HRL 어댑터. 입력 CSV 탐색 함수를 실행기가 사용 |
| `hrl/S37_task_granularity.py` | 비드를 연결 경로 등 작업 단위로 묶음 |
| `hrl/S38_hrl_pointwise.py` | 연결 경로를 판단점 큐로 바꾸고 논문식 이동·대기 환경 구성 |
| `hrl/S39_hrl_paper_training.py` | 상위 배정 PPO와 하위 이동·대기 PPO를 메타 반복으로 학습 |
| `hrl/S40_hrl_pointwise_rapid.py` | 이동·대기 결과를 로봇별 RAPID로 변환 |
| `hrl/S41_execution_metrics.py` | 계획 일정과 RAPID 명령 시간의 대기·가동률 지표 |
| `hrl/S42_hrl_event_time.py` | 로봇별 독립 시간 진행과 곡선 판단점(버전 A) |
| `tools/hrl_event_run.py` | 입력·소스 해시·체크포인트를 보존하며 학습과 평가 실행 |

## 공동 파이프라인에서 불러 쓰는 모듈 (포함하지 않음)

| 모듈 | 쓰는 것 |
|---|---|
| `config` | 속도, 높이 등 공정 상수 |
| `pipeline.S3_path_planning.S31_types` | 비드 로딩, 비드 분류, 로봇 기지, Park 상수 |
| `pipeline.S3_path_planning.S31_analysis` | 비드 군집과 분석 |
| `pipeline.S3_path_planning.S31_cpsat_scheduler` | 셀 좌표계 변환 |
| `pipeline.S3_path_planning.S33_rapid_writer` | RAPID 기본 출력 |
| `pipeline.S3_path_planning.S35_collision_audit` | 3D 캡슐 충돌 감사 |
| `verify.rapid_timeline`, `verify.run_verify` | RAPID 시간축 파싱과 감사 실행 |

## 주요 의존 패키지

`gymnasium`, `stable-baselines3`, `sb3-contrib`(MaskablePPO), `torch`, `numpy`, `matplotlib`

테스트(`tests/`)도 같은 공동 파이프라인 위에서 돌아갑니다. 원 저장소에서는 이 테스트 57개가 모두 통과합니다(2026-11-11 확인).
