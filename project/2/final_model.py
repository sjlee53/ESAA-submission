# -*- coding: utf-8 -*-
"""와인 품질 분류 — 2조 최종 모델

엑스트라 트리(경서) + 랜덤포레스트(희경) + XGBoost(민서)를
소프트 보팅 2:2:1로 결합한다.

전처리와 검증 설정은 common.py를 그대로 쓴다.

코랩에서 쓰는 법
---------------
    !wget -q https://raw.githubusercontent.com/sjlee53/ESAA-submission/main/project/2/common.py
    !wget -q https://raw.githubusercontent.com/sjlee53/ESAA-submission/main/project/2/final_model.py
    from final_model import *

    sub = run_final()          # 검증 + 학습 + 제출 파일 생성
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from common import (load_data, make_x, make_xy, shift_y, unshift_y,
                    get_cv, get_group_cv, make_submission, SEED)


# ------------------------------------------------------------ 개별 모델
# 세 모델 모두 담당자가 튜닝한 끝에 고른 설정이다.
# 엑스트라 트리와 랜덤포레스트는 기본값이 가장 좋았고(30개·10개 조합 탐색),
# XGBoost만 튜닝 효과가 있어 max_depth=8을 쓴다.

def make_et():
    """경서 담당. 기본 설정이 탐색한 10개 조합을 모두 이겼다."""
    return ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)


def make_rf():
    """희경 담당. 30개 조합 중 오차 범위를 넘는 개선이 없어 기본 설정."""
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)


def make_xgb():
    """민서 담당. 432개 조합 탐색 + max_depth를 8까지 올린 설정.

    깊이를 8보다 키워도 나아지지 않는다. 8에서 test의 중복 303행을
    이미 100% 맞히기 때문이다.
    """
    return XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=8,
                         min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                         objective='multi:softprob', eval_metric='mlogloss',
                         random_state=SEED, n_jobs=-1)


def make_final():
    """최종 모델: 소프트 보팅 2:2:1.

    XGBoost는 단독 성능이 가장 낮지만(그룹 CV 0.5594) 예측이 가장 달라
    (엑스트라 트리와 1,000행 중 104행 차이) 섞으면 도움이 된다.
    가중치를 2:2:1로 둔 것은 그 성능 차이를 반영한 것이다.
    """
    return VotingClassifier(
        estimators=[('et', make_et()), ('rf', make_rf()), ('xgb', make_xgb())],
        voting='soft', weights=[2, 2, 1])


# ------------------------------------------------------------ 실행

def run_final(path='submission_final.csv', verbose=True):
    """검증하고, 전체 데이터로 학습하고, 제출 파일을 만든다."""
    train, test, sub = load_data()
    X, y = make_xy(train)
    X_test = make_x(test)

    # XGBoost가 섞여 있으므로 라벨을 0부터 시작하도록 옮긴다
    y_s = shift_y(y)
    model = make_final()

    if verbose:
        print('=== 검증 ===')
        s = cross_val_score(model, X, y_s, cv=get_cv(), n_jobs=1)
        print('일반 CV   %.4f (±%.4f)' % (s.mean(), s.std()))
        print('  폴드별 %s' % np.round(s, 4).tolist())

        cv_g, gid = get_group_cv(train)
        g = cross_val_score(model, X, y_s, cv=cv_g, groups=gid, n_jobs=1)
        print('그룹 CV   %.4f (±%.4f)   <- 처음 보는 성분 조합'
              % (g.mean(), g.std()))
        print()

    model.fit(X, y_s)
    pred = unshift_y(model.predict(X_test)).astype(int)

    print('=== 제출 파일 ===')
    out = make_submission(pred, sub, path=path)

    # test 중 train과 성분이 같은 행을 몇 개나 맞혔는지 확인
    feat = [c for c in train.columns if c not in ('index', 'quality')]
    lookup = train.drop_duplicates(subset=feat).set_index(feat)['quality']
    key = pd.MultiIndex.from_frame(test[feat])
    dup = key.isin(lookup.index)
    if dup.sum():
        hit = (pred[dup] == lookup.reindex(key[dup]).values).mean()
        print()
        print('test %d행 중 train과 성분이 같은 %d행의 정답률: %.1f%%'
              % (len(test), dup.sum(), 100 * hit))
        print('(이 값이 100%%가 아니면 학습이 제대로 안 된 것이다)')

    return out


if __name__ == '__main__':
    run_final()
