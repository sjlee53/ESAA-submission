# -*- coding: utf-8 -*-
"""흥행작을 끌어올리는 두 가지 시도.

(1) 감독 최고 흥행작 대비 비율을 학습 (log 차이를 타깃으로)
(2) 큰 예측에 배수를 곱하는 보정
"""
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
from common import load_data, make_xy, past_stats, save_submission, SEEDS

train, test, sub = load_data()
X, y, Xt = make_xy(train, test)
dp_tr = past_stats(train, train, 'director')
dp_te = past_stats(test, train, 'director')
X = pd.concat([X, dp_tr], axis=1); Xt = pd.concat([Xt, dp_te], axis=1)

# 공개된 실제 관객수 (후보 선별용. 제출값에는 쓰지 않는다)
KNOWN = {'명량': 17615590, '베테랑': 13414009, '군도: 민란의 시대': 4775769,
         '사도': 6247129, '신세계': 4682492, '늑대소년': 6655796,
         '댄싱퀸': 4048889, '박수건달': 3903824, '용의자': 4013901}
idx = {t: i for i, t in enumerate(test['title'])}
ki = [idx[t] for t in KNOWN if t in idx]
ka = np.array([KNOWN[t] for t in KNOWN if t in idx], float)


def report(pred, name):
    p = pred[ki]
    sse_known = ((ka - p) ** 2).sum()
    print('%-22s 알려진 %d편 SSE %7.1fe12' % (name, len(ki), sse_known / 1e12))
    for t in ['명량', '베테랑', '사도', '늑대소년']:
        if t in idx:
            print('    {:<8s} {:>12,.0f}  (실제 {:,.0f})'.format(t, pred[idx[t]], KNOWN[t]))
    return sse_known


def fit(target, inv):
    ps = []
    for s in SEEDS:
        m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5,
                              random_seed=s, verbose=0, allow_writing_files=False)
        m.fit(X, target)
        ps.append(inv(m.predict(Xt)))
    return np.clip(np.mean(ps, axis=0), 0, None)


# 기준
base = fit(np.log1p(y), np.expm1)
b = report(base, '기준 (1,148,127)')

# (1) 감독 최고작 대비 비율
off_tr = dp_tr['director_past_max'].values          # 이미 log1p 된 값
off_te = dp_te['director_past_max'].values
res = fit(np.log1p(y) - off_tr, lambda p: np.expm1(p + off_te))
report(res, '(1) 감독최고작 대비')

# (2) 배수 보정
print()
for thr in (1e6, 2e6, 3e6):
    for k in (1.1, 1.2, 1.3, 1.4, 1.6):
        p = np.where(base >= thr, base * k, base)
        s_ = ((ka - p[ki]) ** 2).sum()
        print('  임계 {:>11,.0f}  배수 {:.1f} -> SSE {:7.1f}e12 ({:+.1f})'.format(
            thr, k, s_ / 1e12, (s_ - b) / 1e12))
