# RLID-NET Project

## Project Overview

### System Name
**RLID-NET (Reinforcement Learning - Low Impact Development Network)**

### Objective
- Optimized allocation of LID(Low Impact Development) for urban flood reduction using RL(Reinforcement Learning)
- SWMM(Storm Water Management Model) based runoff simulation
- Stratigic approach to find LCC(Life Cycle Cost)-runoff multio-bjective LID allocation
---

## System Architecture

### 1. Entire System Structure
```
RLID-NET/
├── src/
│   ├── core/                    # core simulation engine
│   │   ├── swmm_simulator.py   # SWMM simulation management
│   │   └── lid_manager.py      # LID allocation % parameter management
│   ├── rl/                     # RL module
│   │   ├── environment.py      # RL environment manegement
│   │   └── agent.py           # DQN agent implementaion
│   ├── utils/                  # Utility functions
│   │   ├── config.py          # System configuration
│   │   └── visualization.py   # Result visualization
│   └── __init__.py
├── inp_file/                   # SWMM input files
├── results/                    # results
├── main.py                     # main
├── run_batch_training.py       # Batch training
└── test_system.py             # System test
```

### 2. Core Component

#### A. SWMM Simulation (`src/core/swmm_simulator.py`)
- **목적**: 도시 배수 시스템 시뮬레이션 및 데이터 관리
- **의존성**: PySWMM 라이브러리 기반
- **주요 기능**: 
  - 기준선 성능 분석 및 캐싱
  - LID 적용 후 유출량 계산
  - 진행률 표시 지원 시뮬레이션
  - 자동 임시 디렉토리 관리
- **핵심 클래스**:
  - `SubcatchmentData`: 서브캐치먼트 정보 구조체
  - `SimulationResults`: 시뮬레이션 결과 컨테이너
- **주요 메서드**:
  - `get_baseline_performance()`: 기준선 성능 측정 (캐싱)
  - `run_simulation()`: LID 적용 시뮬레이션 (진행률 지원)
  - `find_highest_runoff_subcatchment()`: 최대 유출 지역 자동 식별
  - `get_total_runoff_reduction()`: 유출 저감량 계산

#### B. LID 관리 시스템 (`src/core/lid_manager.py`)
- **목적**: LID 시설 배치, 제약 검증, INP 파일 수정 관리
- **핵심 기능**:
  - 실시간 제약조건 검증 (`can_apply_action`)
  - 자동 INP 파일 생성 및 수정
  - LID 배치/제거 작업 지원
  - SWMM 호환 파라미터 관리
- **주요 클래스**:
  - `LIDType`: 8개 LID 타입 열거형 (SWMM 코드 포함)
  - `LIDPlacement`: LID 배치 정보 구조체
  - `LIDState`: 현재 LID 상태 요약
- **지원 LID 타입** (8개 모두 활성화):
  - **Rain Garden** (RG): 270,000 KRW/m²
  - **Green Roof** (GR): 45,000 KRW/m²
  - **Permeable Pavement** (PP): 110,000 KRW/m²
  - **Infiltration Trench** (IT): 200,000 KRW/m²
  - **Bio-Retention Cell** (BC): 120,000 KRW/m²
  - **Rain Barrel** (RB): 2,000 KRW/m²
  - **Vegetative Swale** (VS): 19,000 KRW/m²
  - **Rooftop Disconnection** (RD): 500 KRW/m²

#### C. 강화학습 환경 (`src/rl/environment.py`)
- **상태 공간**: 10차원 (`EnvironmentState`)
  - LID 면적 비율 (8차원): 불투수면적 대비 정규화된 LID 면적
  - 정규화된 총 비용 (1차원): 동적 예산 대비 0-1 범위
  - 유출 저감율 (1차원): 추정된 0-1 범위
- **액션 공간**: 48개 액션 (8 LID × 6 면적 비율)
  - 면적 비율: [-2%, -1%, 0.5%, 1%, 2%, 3%]
  - 실시간 유효 액션 필터링 지원
- **보상 함수**: Huber Loss 기반
  - WF × CF x 유출저감율 + WC × 비용절약율
  - 환경변수 가중치: `RLID_RUNOFF_WEIGHT`, `RLID_COST_WEIGHT`
- **동적 예산 계산**: `불투수면적 × 최대단위비용 × 0.5`

#### D. DQN 에이전트 (`src/rl/agent.py`)
- **신경망 구조** (`DQN` 클래스): 
  - 입력층: 10 노드 (상태 공간)
  - 은닉층: [64, 32, 16] 노드 (3층 구조)
  - 출력층: 48 노드 (액션 공간)
  - **활성화 함수**: ReLU
  - 드롭아웃: 0.1
  - 가중치 초기화: Xavier Uniform
- **학습 안정화 기법**:
  - Experience Replay Memory (10,000 capacity)
  - Target Network (100 스텝마다 업데이트)
  - Gradient Clipping (max_norm=2.0)
  - Q-value Clamping (0.0-1.2 범위)
- **학습 매개변수**:
  - 학습률: 0.00005 (Adam 옵티마이저)
  - 배치 크기: 64
  - 할인율: 0.99
  - 엡실론 감소: 1.0 → 0.02 (0.995 decay)

#### E. 시각화 및 보고서 (`src/utils/visualization.py`)
- **목적**: 포괄적 훈련 분석 및 전문 보고서 생성
- **핵심 클래스**: `RLIDVisualizer`
- **고급 시각화 기능**:
  - 이동평균 트렌드 분석
  - 다중 축 차트 (보상-유출저감 동시 표시)
  - 로그 스케일 손실 시각화
  - 파이차트 비용 분석
- **Excel 보고서** (openpyxl 기반):
  - 다중 탭 구성 (배치/요약/비용참조)
  - 전문적 스타일링 (헤더 포맷, 자동 열 너비)
  - TOTAL 행 자동 생성
  - 비용 효율성 메트릭 계산
- **생성 파일**: 5개 주요 보고서
  - `reward_trend.png`: 에피소드별 보상 추이
  - `loss_trend.png`: 훈련 손실 추이 (로그 스케일)
  - `training_metrics.xlsx`: 전체 훈련 데이터
  - `lid_placement_summary.xlsx`: LID 배치 최종 결과
  - `baseline_comparison.png`: 4분할 종합 성능 비교

---

## ⚙️ 시스템 설정 현황

### 1. 현재 설정 (`src/utils/config.py`)
```python
# LID 설정 (8개 타입 모두 활성화)
LID_COSTS = {
    'Rain Garden': 270000.0,
    'Green Roof': 45000.0,
    'Permeable Pavement': 110000.0,
    'Infiltration Trench': 200000.0,
    'Bio-Retention Cell': 120000.0,
    'Rain Barrel': 2000.0,
    'Vegetative Swale': 19000.0,
    'Rooftop Disconnection': 500.0
}

# 상태 공간 (10차원)
CURRENT_STATE_SPACE = {
    'current_lid_areas': 8,         # 8개 LID 타입별 면적 비율
    'total_cost_normalized': 1,     # 0-1 정규화된 총 비용
    'runoff_reduction_rate': 1,     # 0-1 유출 저감 비율
}
CURRENT_STATE_SIZE = 10             # 총 상태 공간 크기

# 신경망 설정
NEURAL_NETWORK_CONFIG = {
    'input_size': 10,
    'hidden_layers': [64, 32, 16],
    'output_size': 48,              # 8 LID × 6 면적 비율
    'activation': 'ReLU',
    'dropout_rate': 0.1
}

# 강화학습 설정
class RLConfig:
    num_episodes: int = 150
    max_steps_per_episode: int = int(os.environ.get('RLID_MAX_STEPS', 50))
    learning_rate: float = 0.00005
    batch_size: int = 64
    memory_size: int = 10000
    target_update_freq: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.995
    # 보상 함수 가중치 (환경변수로 설정 가능)
    reward_runoff_weight: float = float(os.environ.get('RLID_RUNOFF_WEIGHT', 0.7))
    reward_cost_weight: float = float(os.environ.get('RLID_COST_WEIGHT', 0.3))
    eval_episodes: int = 5
    save_freq: int = 25
```

### 2. 실험 설정
```python
class ExperimentConfig:
    base_inp_file: str = "inp_file/Example1.inp"
    output_dir: str = "results"
    log_level: str = "INFO"
    target_subcatchments: int = 1  # 단일 서브캐치먼트 집중
    max_lid_area_ratio: float = 0.5  # 최대 50% 불투수 면적
    # 정규화 인자 (동적 계산)
    max_expected_cost: float = 1000000.0  # 예상 최대 비용
    max_expected_runoff: float = 1000.0   # 예상 최대 유출량
```

### 3. 배치 학습 설정 (`run_batch_training.py`)
```python
# 다양한 실험 설정 조합
configurations = [
    # (episodes, steps, runoff_weight, cost_weight, suffix)
    (1000, 25, 0.7, 0.3, "_20250809_1_8lid_w7030"),
    (1000, 40, 0.7, 0.3, "_20250809_1_8lid_w7030"),
    (1000, 50, 0.7, 0.3, "_20250809_1_8lid_w7030"),
    (1000, 40, 0.6, 0.4, "_20250809_1_8lid_w6040"),
    (1000, 40, 0.8, 0.2, "_20250809_1_8lid_w8020"),
    # ... 다양한 조합으로 실험
]
```

---

## 테스트 및 검증

### 1. 시스템 검증 스크립트 (`test_system.py`)
- 모듈 임포트 테스트
- 설정 시스템 검증
- SWMM 분석 테스트
- 빠른 학습 테스트
- 시각화 시스템 테스트

### 2. 최신 테스트 결과
```
테스트 결과 요약:
├── Module Imports: PASS
├── Configuration: PASS
├── SWMM Analysis: PASS
├── Quick Training: PASS
└── Visualization: PASS

전체: 5/5 테스트 통과
```

### 3. 핵심 성능 지표 (2024년 8월 기준)
**기준선 성능:**
- 기준선 유출량: 676.20 m³ (36시간 시뮬레이션)
- 첨두 유출량: 0.142 m³/s
- 대상 서브캐치먼트: ID 5 (15 ha, 최대 유출 기여구역)

**시스템 구성:**
- 상태 벡터 크기: (10,) - 8개 LID 면적 + 2개 정규화 지표
- 액션 공간: 48개 (8 LID × 6 면적 변화율)
- 지원 LID 타입: 8개 모두 활성화

**대표 성능 결과:**
- 최대 유출 저감률: 27.5% (185.9 m³ 저감)
- 비용 효율성: 0.0428 m³/M KRW
- 총 LID 배치 면적: 36,080 m² (불투수 면적의 93.2%)
- 총 설치 비용: 4.86 M KRW

---

## 실행 방법

### 1. 주요 커맨드 라인 인자

| 인자 | 설명 | 기본값 | 사용 예시 |
| --- | --- | --- | --- |
| `--inp_file` | 사용할 SWMM의 `.inp` 파일 경로를 지정합니다. | `inp_file/Example1.inp` | `--inp_file inp_file/inputfile.inp` |
| `--episodes` | 훈련할 에피소드 수를 지정합니다. (`main.py` 전용) | `150` | `--episodes 500` |
| `--output-dir` | 결과 파일이 저장될 디렉토리를 지정합니다. | `./results` | `--output-dir ./my_results` |
| `--quick-test` | 5 에피소드로 빠른 테스트를 실행합니다. (`main.py` 전용) | `False` | `--quick-test` |

### 2. 단일 실험 실행
```bash
# 기본 설정으로 실행
python main.py

# 에피소드 수를 500으로 변경하여 실행
python main.py --episodes 500

# 특정 INP 파일을 지정하여 빠른 테스트 실행
python main.py --quick-test --inp_file inp_file/inputfile.inp
```

### 3. 배치 실험 실행
```bash
# 기본 INP 파일로 배치 실험 실행
python run_batch_training.py

# 특정 INP 파일로 배치 실험 실행
python run_batch_training.py --inp_file inp_file/inputfile.inp

# 참고: 배치 실험의 세부 설정(가중치 등)은 run_batch_training.py 파일 내의 configurations 리스트를 직접 수정해야 합니다.
```

### 4. 시스템 검증
```bash
python test_system.py
```

### 5. 설정 검증
```bash
python -c "from src.utils.config import validate_action_space; validate_action_space()"
```
---

## 개발 이력

### Version History
- **v2.4** (2025년 11월): 시스템 클라우드 탑재 구현
- **v2.3** (2025년 10월): 시뮬레이션 결과 분석
- **v2.2** (2025년 8월): loss 발산을 잡기 위한 cliping 추가
- **v2.1** (2025년 7월): lid 설치에 필요한 SWMM 제약 조건 구현
- **v2.0** (2025년 5월): 초기 시스템  구현
- **v1.0** (2025년 4월): swmm 연동 및 입출력 제어 테스트

## 개발 환경 및 의존성

### 1. 의존성 설치
프로젝트에 필요한 모든 파이썬 라이브러리는 `requirements.txt` 파일에 명시되어 있습니다. 다음 명령어를 사용하여 한 번에 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```

### 2. 필수 라이브러리
```python
# 핵심 의존성
torch                    # 딥러닝
numpy                    # 수치 계산
pyswmm                   # SWMM 시뮬레이션
matplotlib               # 시각화
logging                  # 로깅
pathlib                  # 파일 시스템
dataclasses              # 데이터 구조
```

### 3. 시스템 요구사항
- Python 3.7+
- Windows/Linux 호환
- SWMM 5.1+ 설치 필요