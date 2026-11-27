# -*- coding: utf-8 -*-
"""피처 후보를 한 번에 만들어 제출 파일로 떨어뜨린다. (담당: 다 — 피처)

기준선
------
submission_dirpast_raw.csv = 리더보드 1,148,127
  common.py 의 make_xy() + 감독 과거작 실적(past_stats) + CatBoost

규칙
----
- **test 영화의 실제 관객수를 어떤 식으로도 쓰지 않는다.** 후보를 고르는 데에도 쓰지 않는다.
  (v5_scaled 의 1.2배 보정이 그렇게 오염돼서 폐기했다)
- 한 번에 **하나만 바꾼다.** 두 개를 같이 바꾸면 어느 쪽이 효과인지 알 수 없다.
- **교차검증으로 판단하지 않는다.** 이번 프로젝트에서 CV 가 두 번 다 방향을 거꾸로 알려줬다.
  배급사 정규화는 CV 개선 / LB 43,164 악화, 감독 과거작은 CV 악화 / LB 60,309 개선.
  참고용으로 CV 도 같이 찍지만 판단은 리더보드로 한다.

실행
----
    python feat_variants.py            # 후보 전부 생성
    python feat_variants.py F1 F4      # 일부만 생성
"""
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

from common import load_data, make_xy, past_stats, save_submission, SEEDS, N_SPLITS

HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────── 과거작 실적의 변형
def past_stats_ex(target, source, key='director', years=None, same_genre=False,
                  prefix=None):
    """개봉일이 더 이른 같은 key 의 영화 실적. 조건을 더 걸 수 있다.

    years:      이 연수 안에 개봉한 과거작만 본다 (예: 3 이면 최근 3년)
    same_genre: 장르까지 같은 과거작만 본다
    """
    t = target.reset_index(drop=True)
    s = source.reset_index(drop=True)
    t_day = pd.to_datetime(t['release_time']).values.astype('datetime64[ns]').astype('int64')
    s_day = pd.to_datetime(s['release_time']).values.astype('datetime64[ns]').astype('int64')
    s_key, t_key = s[key].values, t[key].values
    s_gen, t_gen = s['genre'].values, t['genre'].values
    s_val = s['box_off_num'].values.astype(float)
    window = None if years is None else int(years * 365.25 * 24 * 3600 * 1e9)

    out = np.zeros((len(t), 4))
    for i in range(len(t)):
        m = (s_key == t_key[i]) & (s_day < t_day[i])
        if window is not None:
            m &= (s_day >= t_day[i] - window)
        if same_genre:
            m &= (s_gen == t_gen[i])
        v = s_val[m]
        if len(v):
            out[i] = [len(v), v.mean(), v.max(), v[np.argmax(s_day[m])]]

    p = prefix or key
    cols = ['%s_past_n' % p, '%s_past_mean' % p, '%s_past_max' % p, '%s_past_last' % p]
    df = pd.DataFrame(out, columns=cols)
    for c in cols[1:]:
        df[c] = np.log1p(df[c])
    return df


# ─────────────────────────────────────────────── 후보들
# 각 함수는 (X, X_test) 를 받아 바뀐 (X, X_test) 를 돌려준다.

def F0_base(X, Xt, tr, te):
    """기준선. 감독 과거작 네 개(편수·평균·최대·최근작)."""
    return X, Xt


def F1_last_only(X, Xt, tr, te):
    """직전작 하나만 남긴다.

    베테랑의 직전작은 베를린(717만)인데 평균을 쓰면 494만으로 눌린다.
    '그 감독이 마지막으로 뭘 했나'가 더 직접적인 신호일 수 있다.
    """
    drop = ['director_past_mean', 'director_past_max']
    return X.drop(columns=drop), Xt.drop(columns=drop)


def F2_recent3(X, Xt, tr, te):
    """최근 3년 안의 과거작만 본다. 10년 전 흥행은 지금과 무관할 수 있다."""
    a = past_stats_ex(tr, tr, 'director', years=3, prefix='dir3')
    b = past_stats_ex(te, tr, 'director', years=3, prefix='dir3')
    return pd.concat([X, a], axis=1), pd.concat([Xt, b], axis=1)


def F3_same_genre(X, Xt, tr, te):
    """같은 장르의 과거작만 본다. 멜로를 찍던 감독의 액션은 다를 수 있다."""
    a = past_stats_ex(tr, tr, 'director', same_genre=True, prefix='dirg')
    b = past_stats_ex(te, tr, 'director', same_genre=True, prefix='dirg')
    return pd.concat([X, a], axis=1), pd.concat([Xt, b], axis=1)


def F4_dist_past(X, Xt, tr, te):
    """배급사 과거작 실적. 감독에서 통한 방식을 배급사에 적용한다.

    CV 로는 1,551,834 로 나빴지만 CV 를 믿지 않기로 했으므로 한 번 올려 본다.
    """
    a = past_stats_ex(tr, tr, 'distributor', prefix='dist')
    b = past_stats_ex(te, tr, 'distributor', prefix='dist')
    return pd.concat([X, a], axis=1), pd.concat([Xt, b], axis=1)


def F5_season(X, Xt, tr, te):
    """개봉 분기와 요일. 여름·겨울 성수기와 금요일 개봉 관행을 잡아 본다."""
    def mk(df):
        d = pd.to_datetime(df['release_time']).reset_index(drop=True)
        return pd.DataFrame({
            'quarter': d.dt.quarter,
            'dayofweek': d.dt.dayofweek,
            'is_peak': d.dt.month.isin([7, 8, 12, 1]).astype(int),
        })
    return pd.concat([X, mk(tr)], axis=1), pd.concat([Xt, mk(te)], axis=1)


def F6_genre_rating(X, Xt, tr, te):
    """장르 × 관람등급 교차. '청소년 관람불가 액션'과 '전체 관람가 액션'은 다르다."""
    full = pd.concat([tr['genre'] + '_' + tr['screening_rat'],
                      te['genre'] + '_' + te['screening_rat']], ignore_index=True)
    d = pd.get_dummies(full, prefix='gr').astype(int)
    n = len(tr)
    return (pd.concat([X, d.iloc[:n].reset_index(drop=True)], axis=1),
            pd.concat([Xt, d.iloc[n:].reset_index(drop=True)], axis=1))


def F7_ratios(X, Xt, tr, te):
    """규모를 비율로 쪼갠다. 스태프 수 하나로는 대작과 흥행 실패작을 못 가른다."""
    def mk(df):
        df = df.reset_index(drop=True)
        staff = df['num_staff'].astype(float)
        actor = df['num_actor'].astype(float)
        return pd.DataFrame({
            'staff_per_actor': staff / (actor + 1),
            'staff_per_min': staff / df['time'].clip(lower=1),
            'log_staff': np.log1p(staff),
        })
    return pd.concat([X, mk(tr)], axis=1), pd.concat([Xt, mk(te)], axis=1)


CANDIDATES = {
    'F0': (F0_base, '기준선 (감독 과거작만)'),
    'F1': (F1_last_only, '직전작 하나만'),
    'F2': (F2_recent3, '+ 최근 3년 과거작'),
    'F3': (F3_same_genre, '+ 같은 장르 과거작'),
    'F4': (F4_dist_past, '+ 배급사 과거작'),
    'F5': (F5_season, '+ 개봉 분기·요일'),
    'F6': (F6_genre_rating, '+ 장르 x 등급'),
    'F7': (F7_ratios, '+ 규모 비율'),
}


# ─────────────────────────────────────────────── 실행
def cv_score(X, y, seeds=SEEDS):
    """참고용. 판단은 리더보드로 한다."""
    s_ = []
    for s in seeds:
        for a, b in KFold(N_SPLITS, shuffle=True, random_state=s).split(X):
            m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5,
                                  random_seed=s, verbose=0, allow_writing_files=False)
            m.fit(X.iloc[a], np.log1p(y[a]))
            p = np.clip(np.expm1(m.predict(X.iloc[b])), 0, None)
            s_.append(np.sqrt(np.mean((y[b] - p) ** 2)))
    return float(np.mean(s_))


def main():
    want = [a.upper() for a in sys.argv[1:]] or list(CANDIDATES)
    train, test, sub = load_data()
    X0, y, Xt0 = make_xy(train, test)
    X0 = pd.concat([X0, past_stats(train, train, 'director')], axis=1)
    Xt0 = pd.concat([Xt0, past_stats(test, train, 'director')], axis=1)

    print('기준선 피처 {}개\n'.format(X0.shape[1]))
    for name in want:
        fn, desc = CANDIDATES[name]
        X, Xt = fn(X0.copy(), Xt0.copy(), train, test)
        assert list(X.columns) == list(Xt.columns), '열이 어긋났다: ' + name

        preds = []
        for s in SEEDS:
            m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5,
                                  random_seed=s, verbose=0, allow_writing_files=False)
            m.fit(X, np.log1p(y))
            preds.append(np.expm1(m.predict(Xt)))
        pred = np.clip(np.mean(preds, axis=0), 0, None)

        path = os.path.join(HERE, 'submission_%s.csv' % name.lower())
        print('[{}] {}  피처 {}개  (참고 CV {:,.0f})'
              .format(name, desc, X.shape[1], cv_score(X, y)))
        save_submission(pred, sub, path)
        print()


if __name__ == '__main__':
    main()
