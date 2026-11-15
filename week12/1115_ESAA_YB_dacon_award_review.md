# 수상작 리뷰 — 신용카드 사용자 연체 예측 AI 경진대회 (Private 1위)

ESAA YB · 수상작 리뷰 과제 · 이승주

| 항목 | 내용 |
|---|---|
| 대회 | [월간 데이콘 신용카드 사용자 연체 예측 AI 경진대회](https://dacon.io/competitions/official/235713/overview/description) |
| 기간 · 규모 | 2021.04.05 ~ 05.24 · 4,004명 참가 · 상금 100만 원 |
| 과제 유형 | 정형 · 다중분류(3개 클래스) · 확률 제출 |
| 평가 지표 | Log Loss (낮을수록 좋음) |
| 리뷰 대상 | [\[Private 1위 0.6581\] \| 소회의실 \| Catboost](https://dacon.io/competitions/official/235713/codeshare/2768) (조회 25,738 · 추천 42) |
| 최종 성적 | **Private 1위, Log Loss 0.6581** (OOF 교차검증 0.664853) |

작성자 본인이 "logloss를 가장 크게 낮출 수 있었던 요소는 **ID 변수를 만든 것**과 **CatBoost 모델**"이라고 밝혔다. 랜덤 포레스트·LightGBM·XGBoost와 보팅·스태킹을 모두 시도했지만, **파라미터를 건드리지 않은 CatBoost 하나**를 넘지 못했다고 한다.

---

## 1. 데이터

신용카드 사용자의 개인 신상정보로 **신용도 등급(credit)을 예측**하는 문제다.

| 파일 | 크기 | 설명 |
|---|---|---|
| `train.csv` | (26,457, 20) | 신상정보 + 정답 `credit` |
| `test.csv` | (10,000, 19) | 신상정보만 |
| `sample_submission.csv` | (10,000, 4) | `index` + 클래스 0·1·2의 확률 |

타깃 `credit`은 **0·1·2 세 개 등급**이고, 숫자가 낮을수록 신용도가 높다. 제출은 각 등급의 **확률**이며 Log Loss로 채점한다.

입력 변수는 성격별로 이렇게 나뉜다.

- **신상** — `gender`, `car`(차량 소유), `reality`(부동산 소유), `child_num`, `family_size`, `family_type`, `house_type`, `edu_type`
- **소득·직업** — `income_total`, `income_type`, `occyp_type`, `DAYS_EMPLOYED`
- **연락처** — `work_phone`, `phone`, `email`, `FLAG_MOBIL`
- **기간** — `DAYS_BIRTH`(출생일), `begin_month`(카드 발급 월)

데이터 자체에 다루기 까다로운 지점이 셋 있다.

1. `DAYS_BIRTH`·`DAYS_EMPLOYED`·`begin_month`는 **기준일로부터 거슬러 센 음수**다.
2. `DAYS_EMPLOYED`에는 **양수 값**이 섞여 있다 — 고용 일수로는 말이 안 되는 값이고, 실제로는 무직자를 뜻한다.
3. `FLAG_MOBIL`은 **모든 행이 1**이라 정보가 없다.

---

## 2. 코드 흐름

```
결측치·이상치 처리  →  무의미 변수 제거  →  음수/양수 값 정리
      → 파생변수 13개 생성 (핵심: ID)  →  다중공선 변수 제거
      → 로그 변환 · OrdinalEncoder  →  KMeans 군집 변수 추가  →  StandardScaler
      → CatBoost + StratifiedKFold(15) OOF 학습  →  15개 폴드 확률 평균 제출
```

| 단계 | 한 일 |
|---|---|
| 결측치 | 문자열 `'NaN'`으로 채움 (CatBoost가 범주형으로 받도록) |
| 이상치 | `family_size > 7`인 행 제거 |
| 변수 제거 | `index`, `FLAG_MOBIL`(전부 1이라 정보 없음) |
| 값 정리 | `DAYS_EMPLOYED` 양수 → 0 (무직 처리), 음수 변수는 절댓값 |
| 파생변수 | 기간·비율 변수 12개 + **ID 변수 1개** |
| 중복 제거 | 파생변수와 겹치는 `child_num`, `DAYS_BIRTH`, `DAYS_EMPLOYED` 삭제 |
| 인코딩 | `income_total` 로그 변환, 범주형 9개 `OrdinalEncoder` |
| 군집 | KMeans 36개 군집 번호를 변수로 추가 |
| 스케일링 | 수치형 17개 `StandardScaler` |
| 모델 | `CatBoostClassifier()` **기본 파라미터**, 15겹 교차검증, early stopping 100 |
| 제출 | 15개 폴드 모델의 test 확률을 평균 |

---

## 3. 주요 코드 분석

### (1) ID 변수 — 1위를 만든 한 줄

```python
# ID 생성: 각 컬럼의 값들을 더해서 고유한 사람을 파악
# (*한 사람이 여러 개 카드를 만들 가능성을 고려해 begin_month는 제외함)
df['ID'] = \
df['child_num'].astype(str) + '_' + df['income_total'].astype(str) + '_' +\
df['DAYS_BIRTH'].astype(str) + '_' + df['DAYS_EMPLOYED'].astype(str) + '_' +\
df['work_phone'].astype(str) + '_' + df['phone'].astype(str) + '_' +\
df['email'].astype(str) + '_' + df['family_size'].astype(str) + '_' +\
df['gender'].astype(str) + '_' + df['car'].astype(str) + '_' +\
df['reality'].astype(str) + '_' + df['income_type'].astype(str) + '_' +\
df['edu_type'].astype(str) + '_' + df['family_type'].astype(str) + '_' +\
df['house_type'].astype(str) + '_' + df['occyp_type'].astype(str)
```

이 데이터에는 **같은 사람의 행이 여러 번 들어 있다.** 한 사람이 카드를 여러 번 발급받으면 발급 월만 다르고 나머지 신상정보는 전부 같기 때문이다.

보통은 중복을 지우거나 무시하는데, 이 팀은 **신상 정보를 전부 이어 붙여 사람을 식별하는 열쇠로 바꿨다.** 그리고 이 `ID`를 CatBoost의 `cat_features`에 넣어, 모델이 "이 사람은 과거에 어떤 등급을 받았는가"를 학습하도록 했다.

`begin_month`를 **일부러 뺀 것**이 이 설계의 핵심이다. 발급 월까지 넣으면 같은 사람의 서로 다른 카드가 전부 다른 ID가 되어 버려서, 사람을 묶는 기능이 사라진다. 도메인 가정("한 사람이 카드를 여러 번 만들 수 있다")이 코드 한 줄의 선택으로 그대로 들어가 있다.

### (2) 파생변수 — 기간을 여러 축으로 쪼개기

```python
df['before_EMPLOYED'] = df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']   # 취업 전까지의 일수
df['Age'] = df['DAYS_BIRTH'] // 365                              # 나이
df['EMPLOYED'] = df['DAYS_EMPLOYED'] // 365                      # 근속연수
df['DAYS_BIRTH_m'] = np.floor(df['DAYS_BIRTH']/30) - ((np.floor(df['DAYS_BIRTH']/30)/12).astype(int)*12)
df['ability'] = df['income_total'] / (df['DAYS_BIRTH'] + df['DAYS_EMPLOYED'])
df['income_mean'] = df['income_total'] / df['family_size']
```

같은 일수 데이터를 **년·월·주 단위로 나누고, 비율 변수로도 바꿨다.** `ability`(살아온 날 + 일한 날 대비 소득), `income_mean`(가족 1인당 소득)처럼 나눗셈으로 만든 변수가 눈에 띈다.

트리 모델은 뺄셈·나눗셈을 스스로 만들지 못하기 때문에, 이런 변수는 사람이 직접 넣어 줘야 한다. 만든 뒤에 원본 `child_num`, `DAYS_BIRTH`, `DAYS_EMPLOYED`를 **지운 것**도 눈여겨볼 점이다. 파생변수와 겹치는 정보를 남겨 두지 않았다.

### (3) CatBoost — 기본 파라미터, 폴드는 15개

```python
n_fold = 15
skfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
cat_cols = ['income_type','edu_type','family_type','house_type','occyp_type','ID']

for fold in range(n_fold):
    train_data = Pool(data=X_train, label=y_train, cat_features=cat_cols)
    valid_data = Pool(data=X_valid, label=y_valid, cat_features=cat_cols)

    model_cat = CatBoostClassifier()          # 파라미터 기본값
    model_cat.fit(train_data, eval_set=valid_data,
                  use_best_model=True, early_stopping_rounds=100, verbose=100)

    cat_pred[valid_idx] = model_cat.predict_proba(X_valid)
    cat_pred_test += model_cat.predict_proba(X_test) / n_fold
```

세 가지가 인상적이다.

- **파라미터를 건드리지 않았다.** 작성자 설명에 따르면 튜닝한 쪽이 오히려 나빴다.
- **폴드 수를 5부터 17까지 바꿔 보고 15를 골랐다.** 폴드 수를 하이퍼파라미터처럼 다뤘다.
- **범주형을 직접 넘겼다.** `cat_features`에 `ID`를 포함시켜, 원-핫 대신 CatBoost의 범주형 처리를 그대로 썼다. 값의 종류가 수만 개인 `ID`를 원-핫으로 폈다면 차원이 폭발했을 것이다.

폴드별 Log Loss는 **0.6414 ~ 0.6908**로 0.05나 벌어졌고, 전체 OOF는 **0.664853**이었다. 최종 Private는 **0.6581**로 OOF보다 좋았는데, 15개 폴드 모델의 확률을 평균한 효과로 보인다.

---

## 4. 새롭게 알게 된 점 / 아쉬운 점

### 새롭게 알게 된 점

**중복을 지우는 대신 변수로 바꿀 수 있다.** 이번 미니프로젝트(2) 와인 데이터에도 성분이 완전히 같은 행이 858개 있었고, 평가 데이터 1,000행 중 303행이 학습 데이터와 똑같았다. 우리는 그 중복을 **검증을 설계하는 근거**로 썼다 — 같은 와인을 한쪽 폴드에만 넣는 그룹 교차검증. 이 팀은 같은 현상을 **입력 변수**로 썼다. 똑같은 "중복 발견"에서 완전히 다른 두 갈래가 나온 셈이다. 어느 쪽이 맞느냐가 아니라, **중복은 버릴 쓰레기가 아니라 정보**라는 점을 양쪽에서 확인했다.

**도메인 가정이 변수 설계에 직접 들어간다.** `begin_month`를 ID에서 뺀 한 줄이 그렇다. "한 사람이 카드를 여러 번 만들 수 있다"는 생각이 없었으면 전체 열을 다 이어 붙였을 것이고, 그러면 ID는 거의 모든 행이 고유해져 아무 쓸모가 없어진다. 코드는 한 줄이지만 그 앞에 데이터를 들여다본 시간이 있다.

**모델을 바꾸는 쪽이 튜닝보다 크다.** 기본 파라미터 CatBoost가 RF·LGBM·XGB와 보팅·스태킹을 전부 이겼다. 우리도 ExtraTrees·RandomForest 설정 32가지를 돌려서 얻은 것이 +0.0008이었던 반면, 성격이 다른 TabICL을 섞었을 때는 +0.0060이 올랐다. **같은 종류를 깎는 것보다 다른 종류를 가져오는 쪽**이 효율이 좋다는 게 두 대회에서 같은 모양으로 나타났다.

**폴드 수도 조정 대상이다.** 교차검증을 "5겹이면 되는 것"으로만 생각했는데, 5~17을 다 돌려 보고 15를 고른 접근은 처음 봤다. 폴드가 많아지면 폴드당 학습 데이터가 늘어 모델이 좋아지고, 앙상블할 모델 수도 늘어난다.

### 아쉬운 점

**검증이 ID를 고려하지 않는다.** `StratifiedKFold`는 행을 무작위로 섞기 때문에, **같은 사람의 여러 카드가 학습 폴드와 검증 폴드에 나뉘어 들어간다.** `ID`를 변수로 쓰는 모델에서는 검증 폴드의 사람을 학습 때 이미 본 셈이 된다. `StratifiedGroupKFold(groups=ID)`로 다시 재 보면 점수가 꽤 달라졌을 것이다. 우리 와인 프로젝트에서 같은 조치를 했을 때 **0.678에서 0.573으로 10.6%p가 빠졌다.**

다만 이것이 "잘못"은 아니다. **테스트 데이터에도 같은 사람이 들어 있기 때문에** 이 성질을 쓰는 것이 대회에서는 정답이고, 실제로 1위를 했다. 문제는 **교차검증 점수를 모델의 일반화 성능으로 읽을 수 없다**는 것이다. 처음 보는 사람이 들어오면 이 모델은 ID를 못 쓴다. 글에 그 한계에 대한 언급은 없었다.

**KMeans를 스케일링 전에 돌렸다.**

```python
kmeans_train = train.drop(['credit'], axis=1)
kmeans = KMeans(n_clusters=36, random_state=42).fit(kmeans_train)   # 스케일링 전
...
scaler = StandardScaler()                                            # 스케일링은 그 다음
train[numerical_feats] = scaler.fit_transform(train[numerical_feats])
```

이 시점의 `ID`는 0~26,000 범위의 정수다. KMeans는 유클리드 거리로 군집을 나누므로, **ID 하나가 거리 계산을 완전히 지배한다.** 나머지 변수는 사실상 무시되고, 군집은 "ID 번호 구간 나누기"에 가까워진다. 게다가 ID 정수값에는 크기 순서의 의미가 없다(인코딩된 범주일 뿐이다). `n_clusters=36`을 고른 근거도 글에 없다. 스케일링을 먼저 하고, 범주형을 빼고 돌렸어야 한다고 생각한다.

**인코더의 미등장 범주 처리를 확인하지 않았다.** `OrdinalEncoder`는 학습 때 못 본 범주를 전부 한 칸(-1)에 몰아넣는다. **테스트에만 있는 사람은 전부 같은 ID**가 되는데, 그 비율이 얼마나 되는지 점검한 흔적이 없다. ID가 가장 중요한 변수라고 본인도 밝힌 만큼, 이 비율이 성능의 상한을 결정했을 것이다.

**재현이 보장되지 않는다.** `CatBoostClassifier()`에 `random_state`를 주지 않았다. 폴드 분할 시드는 42로 고정했지만 모델 쪽은 비어 있어서, 다시 돌리면 0.6581이 그대로 나온다는 보장이 없다. 우리도 희경님 파이프라인을 재현했을 때 같은 코드에서 0.704~0.709가 나왔던 경험이 있어서, 이 부분이 눈에 들어왔다.

**early stopping을 검증 폴드로 했다.** `eval_set=valid_data` + `use_best_model=True`는 검증 폴드를 보고 반복 횟수를 고르는 것이라, 그 폴드의 OOF 점수는 조금 낙관적이다. 흔히 쓰는 방식이고 영향도 크지 않지만, 보고된 0.664853을 "완전히 깨끗한 검증 점수"로 읽으면 안 된다.

### 우리 프로젝트에 적용한다면

와인 데이터에서 우리는 중복 303행을 **학습 라벨로 덮어쓰는 방식**으로 처리했다. 이 팀의 방식대로라면 **성분 조합을 이어 붙인 ID를 범주형 변수로 넣고 CatBoost에 맡기는 것**도 가능했다. 덮어쓰기는 완전히 일치하는 행만 건지지만, ID를 변수로 주면 모델이 "비슷한 조합"까지 함께 학습할 여지가 생긴다. 다만 와인은 평가 데이터 중복 비율이 30.3%로 이 대회보다 낮을 수 있어, 실제로 이득이 났을지는 돌려 봐야 안다. 다음에 중복이 있는 데이터를 만나면 **지우기 / 검증에서 묶기 / 변수로 만들기** 세 가지를 모두 후보에 올려야겠다.
