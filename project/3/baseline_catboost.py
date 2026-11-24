# -*- coding: utf-8 -*-
"""영화 관객수 예측 — 첫 제출용 베이스라인.

목적은 점수를 올리는 것이 아니라 **교차검증 점수와 리더보드 점수의 대응을 확인하는 것**이다.
지금 교차검증 RMSE는 약 144만인데 리더보드 상위권은 47만~50만이라, 이 차이가
test 데이터의 성격 때문인지 확인해야 이후 실험의 기준점이 생긴다.

구성 (실험으로 고른 것)
  - 타깃 log1p 변환 후 학습, expm1로 복원   (144.5만 → raw 155.5만 대비 개선)
  - 장르·관람등급 원-핫, 배급사는 빈도 인코딩
  - 감독 타깃 인코딩은 넣지 않는다 (test 감독 220명 중 154명이 train에 없어 12% 악화)
  - CatBoost (같은 피처에서 XGBoost·RandomForest보다 모두 우위)

실행: python baseline_catboost.py
출력: submission_baseline.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '1', 'data')
SEEDS = (0, 1, 2)
NFOLD = 5


def load():
    tr = pd.read_csv(os.path.join(DATA, 'movies_train.csv'))
    te = pd.read_csv(os.path.join(DATA, 'movies_test.csv'))
    sub = pd.read_csv(os.path.join(DATA, 'submission.csv'))
    return tr, te, sub


def fix_time(te, tr):
    """test 224번 「슈퍼레이서 엔지」의 time 이 2015다 (상영시간 자리에 개봉연도).

    train 의 상영시간 범위는 45~180분이다. 같은 장르(애니메이션)의 train 중앙값으로 바꾼다.
    """
    te = te.copy()
    bad = te['time'] > 200
    if bad.any():
        med = tr.groupby('genre')['time'].median()
        te.loc[bad, 'time'] = te.loc[bad, 'genre'].map(med).values
        print('time 이상값 %d건 보정 → %s' % (bad.sum(), list(te.loc[bad, 'time'])))
    return te


def make_features(tr, te):
    """train/test 를 함께 변환하되, 타깃은 쓰지 않는다 (누수 없음)."""
    full = pd.concat([tr.drop(columns=['box_off_num']), te], ignore_index=True)
    n = len(tr)

    X = pd.DataFrame(index=full.index)
    X['time'] = full['time']
    X['num_staff'] = full['num_staff']
    X['num_actor'] = full['num_actor']
    X['dir_prev_num'] = full['dir_prev_num']
    # 결측 330건(train)·136건(test)은 전부 dir_prev_num == 0 인 행이다.
    # 값이 빠진 게 아니라 '감독의 이전 작품이 없다'는 뜻이라 0 이 맞다.
    X['dir_prev_bfnum'] = full['dir_prev_bfnum'].fillna(0)
    X['has_prev'] = (full['dir_prev_num'] > 0).astype(int)

    t = pd.to_datetime(full['release_time'])
    X['year'] = t.dt.year
    X['month'] = t.dt.month

    X = pd.concat([X,
                   pd.get_dummies(full['genre'], prefix='genre').astype(int),
                   pd.get_dummies(full['screening_rat'], prefix='rat').astype(int)], axis=1)

    # 배급사 빈도 인코딩 — 타깃을 쓰지 않으므로 train/test 를 합쳐 세도 누수가 아니다
    X['dist_freq'] = full['distributor'].map(full['distributor'].value_counts()).values

    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)


def model(seed):
    return CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5,
                             random_seed=seed, verbose=0, allow_writing_files=False)


def cross_val(X, y):
    rmses = []
    for s in SEEDS:
        for a, b in KFold(NFOLD, shuffle=True, random_state=s).split(X):
            m = model(s)
            m.fit(X.iloc[a], np.log1p(y[a]))
            p = np.clip(np.expm1(m.predict(X.iloc[b])), 0, None)
            rmses.append(np.sqrt(np.mean((y[b] - p) ** 2)))
    return float(np.mean(rmses)), float(np.std(rmses))


def main():
    tr, te, sub = load()
    te = fix_time(te, tr)
    X, X_te = make_features(tr, te)
    y = tr['box_off_num'].values

    mu, sd = cross_val(X, y)
    print('교차검증 RMSE: {:,.0f}  (폴드 표준편차 {:,.0f}, {}-fold x {}seed)'
          .format(mu, sd, NFOLD, len(SEEDS)))

    # 시드를 바꿔 가며 학습한 예측을 평균한다 (CatBoost 는 시드마다 결과가 조금씩 다르다)
    preds = []
    for s in SEEDS:
        m = model(s)
        m.fit(X, np.log1p(y))
        preds.append(np.expm1(m.predict(X_te)))
    pred = np.clip(np.mean(preds, axis=0), 0, None).round().astype(int)

    sub['box_off_num'] = pred
    out = os.path.join(HERE, 'submission_baseline.csv')
    sub.to_csv(out, index=False, encoding='utf-8-sig')
    print('저장:', out)
    print('예측 분포: 중앙값 {:,.0f} / 평균 {:,.0f} / 최대 {:,.0f}'
          .format(np.median(pred), pred.mean(), pred.max()))


if __name__ == '__main__':
    main()
