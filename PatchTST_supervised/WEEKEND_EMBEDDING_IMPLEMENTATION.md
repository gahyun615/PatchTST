# 평일/주말 임베딩 구현 방법 상세 설명

## 전체 프로세스 개요

평일/주말 임베딩은 다음 4단계로 구현됩니다:

1. **데이터 로더**: 시간 정보에서 평일/주말 플래그 추출
2. **패치 변환**: 시퀀스 레벨 플래그를 패치 레벨로 변환
3. **임베딩 레이어**: 학습 가능한 임베딩 테이블 생성
4. **임베딩 결합**: 패치 임베딩과 위치 임베딩에 추가

---

## 1단계: 데이터 로더에서 평일/주말 정보 추출

### 위치: `data_loader.py`의 `Dataset_Custom.__read_data__()`

```python
# Extract weekday information for weekend/weekday embedding
# weekday: 0=Monday, 6=Sunday. Weekend: 5=Saturday, 6=Sunday
df_stamp_weekday = df_raw[["date"]][border1:border2]
df_stamp_weekday["date"] = pd.to_datetime(df_stamp_weekday.date)
df_stamp_weekday["weekday"] = df_stamp_weekday.date.apply(
    lambda row: row.weekday(), 1
)
# 0: weekday (Mon-Fri), 1: weekend (Sat-Sun)
self.weekend_flag = (df_stamp_weekday["weekday"] >= 5).astype(int).values
```

### 설명:
1. **날짜 추출**: 각 시간 스텝의 날짜 정보를 가져옴
2. **요일 계산**: `weekday()` 함수 사용
   - 월요일 = 0, 화요일 = 1, ..., 일요일 = 6
3. **이진 분류**: 
   - `weekday >= 5` → 주말 (토요일=5, 일요일=6) → **1**
   - `weekday < 5` → 평일 (월~금) → **0**
4. **결과**: `self.weekend_flag`는 길이 `seq_len`의 numpy 배열
   - 예: `[0, 0, 0, 0, 0, 1, 1, 0, 0, ...]` (월~금=0, 토일=1)

### `__getitem__()`에서 반환:

```python
# Extract weekend flag for input sequence
seq_x_weekend = self.weekend_flag[s_begin:s_end]

return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_weekend
```

- 입력 시퀀스 길이(`seq_len`)만큼의 평일/주말 플래그를 반환
- 형태: `[seq_len]` 크기의 numpy 배열

---

## 2단계: 패치 단위로 변환

### 위치: `PatchTST_backbone.py`의 `forward()` 메서드

```python
# Process weekend flag for patching if provided
weekend_flag_patched = None
if self.use_weekend_embedding and weekend_flag is not None:
    bs, seq_len = weekend_flag.shape  # weekend_flag: [batch_size, seq_len]
    
    # 패딩 처리 (데이터와 동일하게)
    if self.padding_patch == 'end':
        pad_vals = weekend_flag[:, -1:].repeat(1, self.stride)
        weekend_flag = torch.cat([weekend_flag, pad_vals], dim=1)
    
    # 패치로 변환: [bs x patch_num x patch_len]
    weekend_flag_patched = weekend_flag.unfold(dimension=-1, size=self.patch_len, step=self.stride)
    
    # 평균 기반 결정: 패치 내 시간들의 평균이 0.5보다 크면 주말(1)
    weekend_flag_patched = (weekend_flag_patched.float().mean(dim=-1) > 0.5).long()
    # 결과: [bs x patch_num]
```

### 단계별 설명:

#### 2-1. 패딩 처리
```python
if self.padding_patch == 'end':
    pad_vals = weekend_flag[:, -1:].repeat(1, self.stride)
    weekend_flag = torch.cat([weekend_flag, pad_vals], dim=1)
```
- **목적**: 데이터 패딩과 동일하게 맞춤
- **방법**: 마지막 값을 `stride` 길이만큼 반복
- **예시**: 
  - 원본: `[0, 0, 0, 1, 1]` (seq_len=5)
  - 패딩 후: `[0, 0, 0, 1, 1, 1, 1, 1]` (stride=3인 경우)

#### 2-2. 패치로 변환
```python
weekend_flag_patched = weekend_flag.unfold(dimension=-1, size=self.patch_len, step=self.stride)
```
- **입력**: `[bs, seq_len]` → 예: `[16, 336]`
- **출력**: `[bs, patch_num, patch_len]` → 예: `[16, 41, 16]`
- **동작**: 
  - `patch_len=16`, `stride=8`인 경우
  - 각 패치는 16개 시간 스텝을 포함
  - 8 스텝씩 이동하며 패치 생성

#### 2-3. 평균 기반 결정
```python
weekend_flag_patched = (weekend_flag_patched.float().mean(dim=-1) > 0.5).long()
```
- **입력**: `[bs, patch_num, patch_len]` → 예: `[16, 41, 16]`
- **처리**: 각 패치 내 시간들의 평균 계산
- **결정**: 평균 > 0.5 → 주말(1), 그렇지 않으면 평일(0)
- **출력**: `[bs, patch_num]` → 예: `[16, 41]`

### 예시:
```
패치 내 시간들: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
평균: 10/16 = 0.625 > 0.5
결과: 1 (주말)
```

---

## 3단계: 임베딩 레이어 생성

### 위치: `PatchTST_backbone.py`의 `TSTiEncoder.__init__()`

```python
# Weekend/Weekday embedding (2 classes: weekday=0, weekend=1)
if self.use_weekend_embedding:
    self.weekend_embedding = nn.Embedding(2, d_model)  # 0: weekday, 1: weekend
```

### 설명:
- **`nn.Embedding(2, d_model)`**: 
  - **2**: 임베딩할 클래스 수 (0=평일, 1=주말)
  - **d_model**: 임베딩 차원 (128)
  - **결과**: 2 × 128 = 256개의 학습 가능한 파라미터

### 임베딩 테이블 구조:
```
E[0] = [e₀₀, e₀₁, ..., e₀₁₂₇]  ← 평일 임베딩 벡터 (128차원)
E[1] = [e₁₀, e₁₁, ..., e₁₁₂₇]  ← 주말 임베딩 벡터 (128차원)
```

- 학습 과정에서 이 두 벡터가 최적화됨
- 평일과 주말의 패턴 차이를 인코딩

---

## 4단계: 임베딩 조회 및 결합

### 위치: `PatchTST_backbone.py`의 `TSTiEncoder.forward()`

```python
# Add weekend embedding if provided
if self.use_weekend_embedding and weekend_flag is not None:
    # weekend_flag: [bs x patch_num]
    # Expand to [bs * nvars x patch_num]
    weekend_flag_expanded = weekend_flag.unsqueeze(1).repeat(1, n_vars, 1)  
    # [bs x nvars x patch_num]
    weekend_flag_expanded = weekend_flag_expanded.reshape(-1, weekend_flag.shape[1])  
    # [bs * nvars x patch_num]
    
    # Get embedding: [bs * nvars x patch_num x d_model]
    weekend_emb = self.weekend_embedding(weekend_flag_expanded)
    
    # Add weekend embedding to positional encoding
    u = u + weekend_emb
```

### 단계별 설명:

#### 4-1. 채널 확장
```python
weekend_flag_expanded = weekend_flag.unsqueeze(1).repeat(1, n_vars, 1)
```
- **입력**: `[bs, patch_num]` → 예: `[16, 41]`
- **처리**: 
  - `unsqueeze(1)`: `[16, 1, 41]` (차원 추가)
  - `repeat(1, n_vars, 1)`: `[16, 321, 41]` (변수 수만큼 반복)
- **목적**: 채널 독립 처리와 일관성 유지
- **결과**: 각 변수에 동일한 주말 정보 적용

#### 4-2. Reshape
```python
weekend_flag_expanded = weekend_flag_expanded.reshape(-1, weekend_flag.shape[1])
```
- **입력**: `[bs, n_vars, patch_num]` → 예: `[16, 321, 41]`
- **출력**: `[bs * n_vars, patch_num]` → 예: `[5136, 41]`
- **목적**: 채널 독립 처리 형태로 변환

#### 4-3. 임베딩 조회
```python
weekend_emb = self.weekend_embedding(weekend_flag_expanded)
```
- **입력**: `[bs * n_vars, patch_num]` → 예: `[5136, 41]`
  - 각 값은 0 또는 1 (평일 또는 주말)
- **처리**: 
  - `weekend_flag_expanded[i, j] = 0` → `E[0]` 조회
  - `weekend_flag_expanded[i, j] = 1` → `E[1]` 조회
- **출력**: `[bs * n_vars, patch_num, d_model]` → 예: `[5136, 41, 128]`

#### 4-4. 임베딩 결합
```python
u = u + weekend_emb
```
- **이전**: `u = P_emb + PE_pos` (패치 임베딩 + 위치 임베딩)
- **이후**: `u = P_emb + PE_pos + E_weekend` (주말 임베딩 추가)
- **형태**: `[bs * n_vars, patch_num, d_model]`

---

## 전체 흐름 다이어그램

```
[데이터]
날짜: 2016-07-01 00:00:00 (금요일)
  ↓
[1단계: 데이터 로더]
weekday() → 4 (금요일)
weekday >= 5? → False
weekend_flag = 0
  ↓
[결과] weekend_flag: [0, 0, 0, 0, 0, 1, 1, 0, ...] (seq_len 길이)
  ↓
[2단계: 패치 변환]
패딩 → unfold → 평균 계산
  ↓
[결과] weekend_flag_patched: [0, 0, 0, 1, 1, ...] (patch_num 길이)
  ↓
[3단계: 임베딩 레이어]
nn.Embedding(2, 128)
  ↓
[임베딩 테이블]
E[0] = [평일 벡터 128차원]
E[1] = [주말 벡터 128차원]
  ↓
[4단계: 임베딩 조회 및 결합]
weekend_flag_patched[i] = 0 → E[0] 조회
weekend_flag_patched[i] = 1 → E[1] 조회
  ↓
[최종]
h = P_emb + PE_pos + E_weekend
```

---

## 핵심 포인트

1. **이산적 분류**: 연속값이 아닌 0/1로 분류
2. **패치 단위 집계**: 시퀀스 레벨 → 패치 레벨 변환
3. **학습 가능한 임베딩**: 고정값이 아닌 학습으로 최적화
4. **가산적 결합**: 기존 임베딩에 더하는 방식
5. **채널 독립**: 각 변수에 동일한 주말 정보 적용

---

## 코드 요약

### 전체 파이프라인:

```python
# 1. 데이터 로더
weekend_flag = (weekday >= 5).astype(int)  # [seq_len]

# 2. 패치 변환
weekend_flag_patched = weekend_flag.unfold(...).mean(dim=-1) > 0.5  # [bs, patch_num]

# 3. 임베딩 레이어 (초기화)
self.weekend_embedding = nn.Embedding(2, d_model)

# 4. 임베딩 조회 및 결합
weekend_emb = self.weekend_embedding(weekend_flag_expanded)  # [bs*nvars, patch_num, d_model]
h = P_emb + PE_pos + weekend_emb
```

이렇게 구현하면 모델이 평일과 주말의 패턴 차이를 명시적으로 학습할 수 있습니다!



