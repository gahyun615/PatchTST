# 기존 PatchTST 모델에서의 이론적 변경사항

## 1. 임베딩 공간의 확장

### 기존 PatchTST의 임베딩 구조

기존 PatchTST는 각 패치를 다음과 같이 인코딩합니다:

$$h_i^{(original)} = P_{emb}(x_i) + PE_{pos}(i)$$

여기서:
- $P_{emb}(x_i) \in \mathbb{R}^d$: 패치 $x_i$의 선형 투영 (Linear projection)
- $PE_{pos}(i) \in \mathbb{R}^d$: 패치 위치 $i$에 대한 위치 임베딩 (Positional Encoding)
- $d$: 모델 차원 (d_model = 128)

**이론적 의미**: 
- 임베딩 공간 $\mathcal{H} = \mathbb{R}^d$는 **패치 내용**과 **위치 정보**만으로 구성됨
- 시간적 주기성(periodicity) 정보가 명시적으로 포함되지 않음

### 제안 모델의 임베딩 구조

제안 모델은 임베딩 공간을 다음과 같이 확장합니다:

$$h_i^{(proposed)} = P_{emb}(x_i) + PE_{pos}(i) + E_{weekend}(w_i)$$

여기서:
- $E_{weekend}(w_i) \in \mathbb{R}^d$: 평일/주말 정보 $w_i \in \{0, 1\}$에 대한 학습 가능한 임베딩
- $w_i$: 패치 $i$의 평일/주말 라벨 (0: 평일, 1: 주말)

**이론적 의미**:
- 임베딩 공간이 **시간적 주기성 차원**을 추가로 포함
- $\mathcal{H}' = \mathcal{H} \oplus \mathcal{H}_{weekend}$ (직합 공간)
- 모델이 패치의 **내용**, **위치**, **시간적 주기성**을 동시에 인코딩

## 2. 시간적 주기성 모델링 (Temporal Periodicity Modeling)

### 기존 모델의 한계

기존 PatchTST는 **위치 기반 인코딩**만 사용하므로:
- 위치 $i$와 $i+7$ (1주일 후)의 패치가 **다른 임베딩**을 가짐
- 주기적 패턴(평일 vs 주말)을 **암묵적으로** 학습해야 함
- Attention 메커니즘이 주기성을 발견하는 데 추가 계산 비용 필요

### 제안 모델의 개선

**명시적 주기성 인코딩**:
- 평일 패치들은 $E_{weekend}(0)$을 공유
- 주말 패치들은 $E_{weekend}(1)$을 공유
- 모델이 **명시적으로** 주기적 패턴을 구분 가능

**수학적 표현**:
패치 $i$와 $j$가 같은 요일 타입(평일 또는 주말)을 가지면:
$$E_{weekend}(w_i) = E_{weekend}(w_j)$$

이를 통해 모델은 **요일 타입에 따른 유사성**을 직접 학습할 수 있습니다.

## 3. 정보 이론적 관점 (Information-Theoretic Perspective)

### 상호 정보량 (Mutual Information) 증가

기존 모델:
$$I(Y; X, Pos)$$

제안 모델:
$$I(Y; X, Pos, Weekend) \geq I(Y; X, Pos)$$

**이론적 근거**: 
- 추가 정보(Weekend)는 예측 $Y$에 대한 정보량을 증가시킴
- 데이터 처리 부등식(Data Processing Inequality)에 의해:
  $$I(Y; X, Pos, Weekend) = I(Y; X, Pos) + I(Y; Weekend | X, Pos)$$
- 조건부 상호 정보량 $I(Y; Weekend | X, Pos) > 0$이면 성능 향상 기대

### 엔트로피 감소

예측 분포의 엔트로피:
- 기존: $H(Y|X, Pos)$
- 제안: $H(Y|X, Pos, Weekend) \leq H(Y|X, Pos)$

**의미**: 추가 정보로 인해 예측 불확실성이 감소

## 4. 표현 학습 관점 (Representation Learning)

### 임베딩 공간의 구조

기존 모델의 임베딩 공간:
$$\mathcal{H}_{original} = \{P_{emb}(x) + PE_{pos}(i) : x \in \mathcal{X}, i \in [1, N]\}$$

제안 모델의 임베딩 공간:
$$\mathcal{H}_{proposed} = \{P_{emb}(x) + PE_{pos}(i) + E_{weekend}(w) : x \in \mathcal{X}, i \in [1, N], w \in \{0,1\}\}$$

### 클러스터링 효과

임베딩 공간에서:
- **평일 패치들**: $E_{weekend}(0)$으로 인해 유사한 방향으로 이동
- **주말 패치들**: $E_{weekend}(1)$으로 인해 다른 방향으로 이동

**결과**: 임베딩 공간이 **요일 타입에 따라 구조화**됨

### 학습 효율성

**기존 모델**:
- Attention이 주기성을 발견하기 위해 많은 학습 필요
- 긴 시퀀스에서 장기 의존성 학습 어려움

**제안 모델**:
- 주기성이 **명시적으로 인코딩**되어 학습 효율 향상
- Attention은 주기성 외의 **복잡한 패턴**에 집중 가능

## 5. 수학적 정식화

### 기존 모델의 Forward Pass

$$\begin{align}
P_i &= W_P \cdot x_i \quad \text{(Patch embedding)} \\
h_i &= P_i + PE_{pos}(i) \quad \text{(Input to Transformer)} \\
z_i &= \text{Transformer}(h_i) \quad \text{(Encoder output)}
\end{align}$$

### 제안 모델의 Forward Pass

$$\begin{align}
P_i &= W_P \cdot x_i \quad \text{(Patch embedding)} \\
w_i &= \text{MajorityVote}(\{w_t : t \in \text{patch}_i\}) \quad \text{(Patch-level weekend flag)} \\
h_i &= P_i + PE_{pos}(i) + E_{weekend}(w_i) \quad \text{(Input to Transformer)} \\
z_i &= \text{Transformer}(h_i) \quad \text{(Encoder output)}
\end{align}$$

### 차이점

**핵심 변경**: 
$$h_i^{(new)} = h_i^{(old)} + E_{weekend}(w_i)$$

이는 **가산적 모듈(Additive Module)**로 구현되어:
- 기존 아키텍처와의 **호환성** 유지
- **최소한의 파라미터 증가** (2 × d_model = 256 parameters)
- **그래디언트 흐름**에 영향 최소화

## 6. 이론적 장점 요약

1. **표현력 증가**: 임베딩 공간에 시간적 주기성 차원 추가
2. **학습 효율성**: 명시적 주기성 인코딩으로 학습 속도 향상
3. **일반화 성능**: 주기적 패턴을 명시적으로 모델링하여 일반화 개선
4. **해석 가능성**: 평일/주말 임베딩이 학습된 값으로 패턴 분석 가능
5. **계산 효율성**: Attention이 주기성 외 패턴에 집중하여 효율적 학습

## 7. 잠재적 한계 및 고려사항

1. **고정된 주기성**: 7일 주기만 모델링 (다른 주기성은 학습 필요)
2. **패치 단위 집계**: 패치 내 시간들의 평균으로 정보 손실 가능
3. **데이터 의존성**: 주기성이 없는 데이터셋에서는 효과 제한적

## 8. 확장 가능성

이론적으로 동일한 방식으로 확장 가능:
- **계절성 임베딩**: 월별, 계절별 패턴
- **시간대 임베딩**: 오전/오후, 시간대별 패턴
- **이벤트 임베딩**: 공휴일, 특별 이벤트 등

$$h_i = P_{emb}(x_i) + PE_{pos}(i) + E_{weekend}(w_i) + E_{season}(s_i) + E_{hour}(h_i) + \cdots$$

이는 **모듈러 설계**로 다양한 시간적 특성을 독립적으로 추가 가능함을 의미합니다.









