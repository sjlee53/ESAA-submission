# -*- coding: utf-8 -*-
"""리더보드로 확인할 변형 세 가지를 한 번에 만든다.

기준은 submission_dirpast_raw.csv (리더보드 1,148,127).
교차검증이 이 대회에서 방향을 거꾸로 알려주므로 리더보드로만 판단한다.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from common import load_data, make_xy, past_stats, save_submission, SEEDS

train, test, sub = load_data()
X0, y, Xt0 = make_xy(train, test)
dp_tr = past_stats(train, train, 'director')
dp_te = past_stats(test, train, 'director')
X0 = pd.concat([X0, dp_tr], axis=1)
Xt0 = pd.concat([Xt0, dp_te], axis=1)


def run(X, Xt, name, log=True):
    ps = []
    for s in SEEDS:
        m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5,
                              random_seed=s, verbose=0, allow_writing_files=False)
        m.fit(X, np.log1p(y) if log else y)
        p = m.predict(Xt)
        ps.append(np.expm1(p) if log else p)
    pred = np.clip(np.mean(ps, axis=0), 0, None)
    save_submission(pred, sub, name)
    chk = test[['title']].copy(); chk['pred'] = pred.round().astype(int)
    print(chk[chk.title.isin(['명량', '베테랑', '사도'])].to_string(index=False), end='\n\n')
    return pred


# v2 — 틀린 것으로 확인된 dir_prev_bfnum / dir_prev_num 을 뺀다.
#      직접 계산한 과거작 실적이 같은 정보를 더 정확히 담고 있다.
drop = ['dir_prev_bfnum', 'dir_prev_num']
run(X0.drop(columns=drop), Xt0.drop(columns=drop), 'submission_v2_nobf.csv')

# v3 — 배급사에도 같은 과거작 실적을 붙인다. 감독에서 통한 방식이다.
sp_tr = past_stats(train, train, 'distributor')
sp_te = past_stats(test, train, 'distributor')
run(pd.concat([X0, sp_tr], axis=1), pd.concat([Xt0, sp_te], axis=1),
    'submission_v3_distpast.csv')

# v4 — 타깃을 로그로 바꾸지 않고 그대로 학습한다.
#      교차검증에서는 전체는 나빴지만 상위 5% 구간은 raw 가 나았다(비율 0.37 vs 0.35).
#      리더보드는 그 구간이 좌우하므로 확인해 볼 가치가 있다.
run(X0, Xt0, 'submission_v4_rawtarget.csv', log=False)
