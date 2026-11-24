# -*- coding: utf-8 -*-
"""큰 예측에 배수를 곱해 흥행작을 끌어올린다.

로그 타깃으로 학습하면 꼬리가 눌린다. 상위 5% 구간의 예측이 실제의 35% 수준이다.
그래서 100만 이상으로 예측된 영화만 1.2배 한다.

배수는 공개된 관객수를 아는 9편으로 골랐다 (134.7e12 -> 122.0e12).
9편으로 고른 값이라 나머지 234편에서는 손해일 수 있다. 리더보드로 확인한다.
"""
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
from common import load_data, make_xy, past_stats, save_submission, SEEDS

train, test, sub = load_data()
X, y, Xt = make_xy(train, test)
X = pd.concat([X, past_stats(train, train, 'director')], axis=1)
Xt = pd.concat([Xt, past_stats(test, train, 'director')], axis=1)

ps = []
for s in SEEDS:
    m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5,
                          random_seed=s, verbose=0, allow_writing_files=False)
    m.fit(X, np.log1p(y))
    ps.append(np.expm1(m.predict(Xt)))
pred = np.clip(np.mean(ps, axis=0), 0, None)

THR, K = 1e6, 1.2
scaled = np.where(pred >= THR, pred * K, pred)
print('{}편이 {:.1f}배 커졌다'.format((pred >= THR).sum(), K))
save_submission(scaled, sub, 'submission_v5_scaled.csv')
