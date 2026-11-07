# -*- coding: utf-8 -*-
"""와인 품질 분류 — 2조 공통 코드

11/6 회의에서 정한 전처리와 검증 설정을 한곳에 모았다.
세 모델 담당자는 이 파일을 그대로 쓴다. 전처리를 각자 다르게 하면 점수를 비교할 수 없다.

코랩에서 쓰는 법
---------------
    !wget -q https://raw.githubusercontent.com/sjlee53/ESAA-submission/main/project/2/common.py
    from common import *

    train, test, sub = load_data()
    X, y = make_xy(train)
    X_test = make_x(test)

    from sklearn.ensemble import ExtraTreesClassifier
    evaluate(ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1), X, y)
"""

import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     cross_val_score)

DATA_URL = 'https://bit.ly/3i4n1QB'      # DACON 와인 품질 분류 데이터
SEED = 42
N_SPLITS = 5


# ---------------------------------------------------------------- 데이터

def load_data(folder='data'):
    """train, test, sample_submission을 읽는다. 없으면 받아온다."""
    if not os.path.exists(os.path.join(folder, 'train.csv')):
        urllib.request.urlretrieve(DATA_URL, 'wine.zip')
        with zipfile.ZipFile('wine.zip') as z:
            z.extractall(folder)

    train = pd.read_csv(os.path.join(folder, 'train.csv'))
    test = pd.read_csv(os.path.join(folder, 'test.csv'))
    sub = pd.read_csv(os.path.join(folder, 'sample_submission.csv'))
    return train, test, sub


# ---------------------------------------------------------------- 전처리

def make_x(df):
    """11/6 회의에서 정한 전처리. 이 네 줄이 전부다.

    - index 제거
    - type -> is_white (white=1, red=0)
    - 결합 이산화황 = 총 이산화황 - 유리 이산화황
    - 그 외에는 아무것도 하지 않는다
      (로그 변환 없음, 이상치 처리 없음, 중복 제거 없음, 변수 제거 없음)
    """
    X = df.drop(columns=[c for c in ('index', 'quality') if c in df.columns])
    X['is_white'] = (X['type'] == 'white').astype(int)
    X = X.drop(columns=['type'])
    X['bound_so2'] = X['total sulfur dioxide'] - X['free sulfur dioxide']
    return X


def make_xy(train):
    """학습용 X와 y를 돌려준다."""
    return make_x(train), train['quality']


def shift_y(y):
    """XGBoost용. 라벨이 0부터 시작해야 해서 3~9를 0~6으로 옮긴다.

        model.fit(X, shift_y(y))
        pred = unshift_y(model.predict(X_test))
    """
    return y - 3


def unshift_y(pred):
    """shift_y로 학습한 모델의 예측을 다시 3~9로 되돌린다."""
    return np.asarray(pred) + 3


# ---------------------------------------------------------------- 검증

def get_cv():
    """공통 교차검증 설정. 모두 이것만 쓴다."""
    return StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)


def get_group_cv(train):
    """최종 후보 확인용. 성분이 같은 와인을 한 폴드에 묶는다.

        cv_g, groups = get_group_cv(train)
        cross_val_score(model, X, y, cv=cv_g, groups=groups)
    """
    feat = [c for c in train.columns if c not in ('index', 'quality')]
    groups = train.groupby(feat, sort=False).ngroup()
    return StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED), groups


def evaluate(model, X, y, name='', shift=False, verbose=True):
    """공통 설정으로 교차검증하고 평균과 폴드별 점수를 돌려준다.

    shift=True 면 XGBoost용으로 라벨을 옮겨서 평가한다.
    """
    target = shift_y(y) if shift else y
    scores = cross_val_score(model, X, target, cv=get_cv(), n_jobs=1)
    if verbose:
        label = name or model.__class__.__name__
        print('%-26s %.4f (±%.4f)' % (label, scores.mean(), scores.std()))
        print('  폴드별 %s' % np.round(scores, 4).tolist())
    return scores.mean(), scores


def evaluate_group(model, X, y, train, name='', shift=False):
    """최종 후보용. 처음 보는 성분 조합에 대한 성능을 본다.

    일반 교차검증보다 10%p 가량 낮게 나오는 것이 정상이다.
    중복된 와인이 학습과 검증에 나뉘어 들어가지 않기 때문이다.
    """
    cv_g, groups = get_group_cv(train)
    target = shift_y(y) if shift else y
    scores = cross_val_score(model, X, target, cv=cv_g, groups=groups, n_jobs=1)
    print('%-26s %.4f (±%.4f)  [그룹 검증]'
          % (name or model.__class__.__name__, scores.mean(), scores.std()))
    return scores.mean(), scores


BASELINE = 0.4395   # 전부 6점으로 예측했을 때


# ---------------------------------------------------------------- 제출

def make_submission(pred, sub, path='submission.csv'):
    """제출 파일을 만들고 형식을 검사한다."""
    pred = np.asarray(pred)
    out = sub.copy()
    out['quality'] = pred

    problems = []
    if list(out.columns) != ['index', 'quality']:
        problems.append('열 이름이 [index, quality]가 아님: %s' % list(out.columns))
    if len(out) != 1000:
        problems.append('행이 1000개가 아님: %d' % len(out))
    if not np.issubdtype(out['quality'].dtype, np.integer):
        problems.append('quality가 정수형이 아님: %s' % out['quality'].dtype)
    lo, hi = out['quality'].min(), out['quality'].max()
    if lo < 3 or hi > 9:
        problems.append('quality가 3~9 범위를 벗어남: %d ~ %d '
                        '(XGBoost면 +3을 빠뜨렸는지 확인)' % (lo, hi))

    if problems:
        print('[제출 파일 점검] 문제 발견')
        for p in problems:
            print('  - ' + p)
    else:
        out.to_csv(path, index=False)
        print('[제출 파일 점검] 이상 없음 -> %s 저장' % path)
        print(out['quality'].value_counts().sort_index().to_string())
    return out
