# -*- coding: utf-8 -*-
"""영화 관객수 예측 — 2조 공통 코드

11/24 회의에서 정한 전처리와 검증 설정을 한곳에 모았다.
**모델 담당자는 이 파일을 그대로 쓴다.** 전처리를 각자 다르게 하면 점수를 비교할 수 없다.

코랩에서 쓰는 법
---------------
    !wget -q https://raw.githubusercontent.com/sjlee53/ESAA-submission/YB/project/3/common.py
    from common import *

    train, test, sub = load_data()
    X, y, X_test = make_xy(train, test)

    from catboost import CatBoostRegressor
    evaluate(CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5,
                               verbose=0, allow_writing_files=False), X, y)

제출까지
-------
    pred = fit_predict(model, X, y, X_test)
    save_submission(pred, sub, 'submission_내이름.csv')

규칙 요약
--------
    dir_prev_bfnum 결측  -> 0 으로 채우고 has_prev 추가
    test time = 2015     -> 장르 중앙값(애니메이션 83분)으로 교체
    198분·40분, dir_prev_num 6 -> 그대로 둔다 (범위 밖일 뿐 오류가 아니다)
    num_staff = 0        -> 그대로 둔다 (소규모 영화 신호)
    감독                 -> 쓰지 않는다 (test 63% 미등장, 타깃 인코딩 시 RMSE 12% 악화)
    배급사               -> 표기 정규화 후 빈도 인코딩
    장르·관람등급         -> 원-핫
    타깃                 -> log1p 로 학습하고 expm1 로 되돌린다
"""

import os
import re
import urllib.request

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

RAW = 'https://raw.githubusercontent.com/sjlee53/ESAA-submission/YB/project/1/data/'
SEED = 42
N_SPLITS = 5
SEEDS = (0, 1, 2)          # 교차검증·제출 모두 이 세 개를 평균한다

__all__ = ['load_data', 'clean', 'make_xy', 'make_x', 'rmse', 'evaluate',
           'fit_predict', 'save_submission', 'SEED', 'N_SPLITS', 'SEEDS']


# ---------------------------------------------------------------- 데이터

def load_data(folder=None):
    """train, test, submission 을 읽는다. 로컬에 없으면 깃허브에서 받아온다."""
    if folder is None:
        here = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
        folder = os.path.join(here, '..', '1', 'data')

    def read(name):
        p = os.path.join(folder, name)
        return pd.read_csv(p) if os.path.exists(p) else pd.read_csv(RAW + name)

    return read('movies_train.csv'), read('movies_test.csv'), read('submission.csv')


# ---------------------------------------------------------------- 전처리

def _norm_distributor(s):
    """`(주)NEW` 와 `NEW`, `시네마달`·`시네마 달`·`(주)시네마달` 을 같은 값으로 만든다.

    이렇게 정리하면 고유 배급사가 204 -> 191 로 줄고,
    같은 회사의 작품 수가 합쳐져 빈도 인코딩이 정확해진다.
    """
    s = re.sub(r'\(주\)|㈜|주식회사', '', str(s))
    return re.sub(r'[\s\-\.]', '', s).strip()


def clean(df, time_median=None):
    """원본을 건드리지 않고 정리본을 돌려준다.

    time_median: 장르별 상영시간 중앙값 (train 에서 구한 것). None 이면 보정하지 않는다.
    """
    df = df.copy()

    # 상영시간 입력 오류 — test 「슈퍼레이서 엔지」가 2015분(개봉연도)으로 적혀 있다.
    # 198분·40분은 train 범위(45~180) 밖이지만 실제로 가능한 값이라 건드리지 않는다.
    if time_median is not None:
        bad = df['time'] > 200
        if bad.any():
            df.loc[bad, 'time'] = df.loc[bad, 'genre'].map(time_median).values

    # 결측 100%가 dir_prev_num == 0 인 행이다. 값이 빠진 게 아니라 '이전작이 없다'는 뜻.
    df['has_prev'] = (df['dir_prev_num'] > 0).astype(int)
    df['dir_prev_bfnum'] = df['dir_prev_bfnum'].fillna(0)

    df['dist_norm'] = df['distributor'].map(_norm_distributor)
    return df


def make_xy(train, test):
    """학습용 X, y 와 test 용 X_test 를 함께 만든다.

    타깃을 쓰지 않는 변환만 하므로 train/test 를 함께 처리해도 누수가 아니다.
    (타깃 인코딩이나 스케일러를 쓰고 싶다면 반드시 폴드 안에서 fit 해야 한다.)
    """
    time_median = train.groupby('genre')['time'].median()
    tr = clean(train, time_median)
    te = clean(test, time_median)

    y = tr['box_off_num'].values
    full = pd.concat([tr.drop(columns=['box_off_num']), te], ignore_index=True)
    n = len(tr)

    X = pd.DataFrame(index=full.index)
    X['time'] = full['time']
    X['num_staff'] = full['num_staff']           # 0 은 소규모 영화 신호라 그대로 둔다
    X['num_actor'] = full['num_actor']
    X['dir_prev_num'] = full['dir_prev_num']
    X['dir_prev_bfnum'] = full['dir_prev_bfnum']
    X['has_prev'] = full['has_prev']

    t = pd.to_datetime(full['release_time'])
    X['year'] = t.dt.year
    X['month'] = t.dt.month

    X = pd.concat([X,
                   pd.get_dummies(full['genre'], prefix='genre').astype(int),
                   pd.get_dummies(full['screening_rat'], prefix='rat').astype(int)], axis=1)

    # 배급사는 '어느 회사인가'보다 '몇 편을 낸 회사인가'가 신호다.
    # 감독은 test 63% 가 train 에 없어서 쓰지 않는다.
    X['dist_freq'] = full['dist_norm'].map(full['dist_norm'].value_counts()).values

    return (X.iloc[:n].reset_index(drop=True),
            y,
            X.iloc[n:].reset_index(drop=True))


def make_x(train, test):
    """X_test 만 필요할 때."""
    return make_xy(train, test)[2]


# ---------------------------------------------------------------- 평가

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def evaluate(model, X, y, log=True, seeds=SEEDS, verbose=True):
    """5-fold 교차검증을 seeds 개수만큼 반복해 RMSE 평균을 낸다.

    log=True 면 log1p 로 학습하고 expm1 로 되돌린 뒤 원래 스케일에서 평가한다
    (대회 지표가 원래 스케일의 RMSE 이므로).

    반환: (평균 RMSE, 폴드별 RMSE 표준편차)
    """
    scores = []
    for s in seeds:
        for tr_i, va_i in KFold(N_SPLITS, shuffle=True, random_state=s).split(X):
            m = clone(model) if hasattr(model, 'get_params') else model
            _set_seed(m, s)
            yt = np.log1p(y[tr_i]) if log else y[tr_i]
            m.fit(X.iloc[tr_i], yt)
            p = m.predict(X.iloc[va_i])
            if log:
                p = np.expm1(p)
            scores.append(rmse(y[va_i], np.clip(p, 0, None)))
    mu, sd = float(np.mean(scores)), float(np.std(scores))
    if verbose:
        name = model.__class__.__name__
        print('{:<24s} RMSE {:>12,.0f}   (폴드 표준편차 {:>9,.0f}, {}-fold x {}seed)'
              .format(name, mu, sd, N_SPLITS, len(seeds)))
    return mu, sd


def _set_seed(m, s):
    for key in ('random_state', 'random_seed', 'seed'):
        try:
            if key in m.get_params():
                m.set_params(**{key: s})
                return
        except Exception:
            pass


def fit_predict(model, X, y, X_test, log=True, seeds=SEEDS):
    """시드를 바꿔 가며 전체 train 으로 학습한 예측을 평균한다."""
    preds = []
    for s in seeds:
        m = clone(model) if hasattr(model, 'get_params') else model
        _set_seed(m, s)
        m.fit(X, np.log1p(y) if log else y)
        p = m.predict(X_test)
        preds.append(np.expm1(p) if log else p)
    return np.clip(np.mean(preds, axis=0), 0, None)


def save_submission(pred, sub, path='submission.csv'):
    """제출 파일을 쓴다. 행 순서는 test 와 같다."""
    out = sub.copy()
    out['box_off_num'] = np.asarray(pred).round().astype(int)
    out.to_csv(path, index=False, encoding='utf-8-sig')
    print('저장: {}  (중앙값 {:,.0f} / 평균 {:,.0f} / 최대 {:,.0f})'
          .format(path, np.median(out.box_off_num), out.box_off_num.mean(),
                  out.box_off_num.max()))
    return out


# ---------------------------------------------------------------- 확인용

if __name__ == '__main__':
    train, test, sub = load_data()
    X, y, X_test = make_xy(train, test)
    print('X {}  y {}  X_test {}'.format(X.shape, y.shape, X_test.shape))
    print('열:', list(X.columns))
    assert X.columns.equals(X_test.columns), '열이 어긋났다'
    assert X.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0, '결측이 남았다'
    assert X_test['time'].max() <= 200, 'time 보정이 안 됐다'
    print('\n기준 모델 (이 점수를 넘기는 것이 목표):')
    from catboost import CatBoostRegressor
    evaluate(CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5,
                               verbose=0, allow_writing_files=False), X, y)
