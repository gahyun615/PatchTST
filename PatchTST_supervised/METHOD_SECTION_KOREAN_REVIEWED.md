# Method Section (한국어 - 검토 완료)

## 1. PatchTST 구조

본 연구는 시계열 예측을 위해 PatchTST 모델을 기반으로 한다.

PatchTST는 입력 시계열을 일정 길이의 패치(patch) 단위로 분할한 뒤, 이를 Transformer Encoder에 입력하여 장기 의존성을 효율적으로 학습한다.

주요 특징은 다음과 같다.

### Patch 기반 처리 (Patch-based Processing)
- 입력 시계열을 고정 길이(patch_len)의 패치로 나누고, stride 간격으로 이동하며 패치를 생성한다.

### 채널 독립 처리 (Channel Independence)
- 전력 데이터의 각 변수(전력 계량기/피처)는 독립적으로 처리된다.

### Transformer Encoder 사용
- Multi-Head Self-Attention을 통해 시계열의 장기 패턴을 학습한다.

### RevIN (Reversible Instance Normalization)
- 입력 데이터를 정규화한 뒤, 예측 결과를 다시 원래 스케일로 복원하여 일반화 성능을 향상시킨다.

## 2. 평일 / 주말 임베딩 (Weekend/Weekday Embedding)

기존 PatchTST는 시간 정보(batch_x_mark)를 활용하지 않기 때문에, 본 연구에서는 평일/주말 정보를 임베딩 형태로 모델에 추가하였다.

### (1) 평일/주말 정보 추출
- 각 시간의 date 값에서 `weekday()`를 추출
- 요일을 다음과 같이 이진 분류:
  - **평일(Weekday)**: 월~금 → 0
  - **주말(Weekend)**: 토~일 → 1
- 이를 통해 각 시간 스텝마다 $w_i \in \{0, 1\}$의 플래그를 생성한다.

### (2) 패치 단위로 변환 (Patch-level Aggregation)
PatchTST가 패치 단위로 작동하기 때문에, 평일/주말 정보도 패치 단위로 변환하였다.

- 하나의 패치에 포함된 시간 구간의 weekday 값들을 모음
- **평균 기반 결정**: 해당 패치 내 시간들의 평균값을 계산하고, 0.5보다 크면 주말(1), 그렇지 않으면 평일(0)로 결정
  $$\bar{w}_i = \begin{cases} 1 & \text{if } \frac{1}{patch\_len}\sum_{t}^{t+patch\_len-1} w_t > 0.5 \\ 0 & \text{otherwise} \end{cases}$$
- 패딩이 필요한 경우, 마지막 값으로 stride 길이만큼 패딩하여 패치 수를 일치시킴
- 이를 통해 patch 단위의 평일/주말 벡터를 만든다.

### (3) 임베딩 결합 방식
모델 내부에서 평일/주말 정보는 학습 가능한 임베딩 벡터로 변환된다.

- **임베딩 함수**: $E: \{0, 1\} \rightarrow \mathbb{R}^d$
- **임베딩 차원**: d_model = 128
- **채널 확장**: 각 변수에 대해 동일한 weekend embedding을 적용 (채널 독립 처리와 일관성 유지)

최종적으로 Transformer 입력 벡터는 다음과 같이 구성된다:

$$h = P_{emb} + PE_{pos} + E_{weekend}$$

- $P_{emb}$: 패치 임베딩
- $PE_{pos}$: 위치 임베딩 (Positional Encoding)
- $E_{weekend}$: 평일/주말 임베딩

이를 통해 모델은 같은 패턴이라도 평일과 주말을 구분하여 학습할 수 있게 된다.

## 3. 실험 설정 (Experimental Settings)

본 연구는 Electricity 데이터셋에 대해 다음과 같은 실험 설정을 사용하였다.

### 3.1 데이터셋 설정

| 항목 | 값 |
|------|-----|
| 데이터셋 | Electricity |
| 변수 수 (Features) | 321 |
| 입력 길이 (seq_len) | 336 |
| 예측 길이 (pred_len) | 96, 192, 336, 720 |
| 데이터 분할 | Train: 70%, Val: 10%, Test: 20% |
| 특징 모드 | Multivariate (M) |

### 3.2 모델 하이퍼파라미터

| 하이퍼파라미터 | 값 | 설명 |
|---------------|-----|------|
| Encoder layers (e_layers) | 3 | Transformer Encoder 레이어 수 |
| Attention heads (n_heads) | 16 | Multi-head attention 헤드 수 |
| Model dimension (d_model) | 128 | 모델 임베딩 차원 |
| Feed-forward dimension (d_ff) | 256 | Feed-forward 네트워크 차원 |
| Dropout | 0.2 | 일반 dropout 비율 |
| FC dropout | 0.2 | Fully connected layer dropout |
| Head dropout | 0.0 | Head layer dropout |
| Patch length (patch_len) | 16 | 패치 길이 |
| Stride | 8 | 패치 생성 stride |
| Padding patch | end | 패치 패딩 방식 |
| RevIN | True | Reversible Instance Normalization 사용 |
| Weekend embedding | True | 평일/주말 임베딩 사용 |

### 3.3 학습 설정

| 항목 | 값 |
|------|-----|
| Batch size | 16 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Learning rate scheduler | OneCycleLR (pct_start=0.2) |
| Training epochs | 100 |
| Early stopping patience | 10 |
| Random seed | 2021 |

---

## 주요 수정 사항

1. **"다수결" → "평균 기반 결정"**: 코드에서는 실제로 평균값이 0.5보다 큰지로 판단하므로 더 정확한 표현으로 수정
2. **패딩 처리 추가**: 패치 수를 맞추기 위한 패딩 과정 명시
3. **채널 확장 설명 추가**: 채널 독립 처리와의 일관성을 위한 설명 추가
4. **수식 정확성**: 수식을 코드 로직과 일치하도록 수정

