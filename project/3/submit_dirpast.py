# -*- coding: utf-8 -*-
"""감독 과거작 실적 피처를 넣은 제출 파일을 만든다.

교차검증은 나빠진다고 했지만(1,438,083 -> 1,473,722) 리더보드는 반대였다.
같은 기반에서 1,245,912 -> 1,185,603 으로 60,309 좋아졌다.
교차검증의 흥행작 30편과 test 의 흥행작이 다른 영화라서 생기는 차이다.

배급사 표기 정규화는 빼고(norm_dist=False) 쓴다. 리더보드에서 43,164 나빴다.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from common import load_data, make_xy, past_stats, SEEDS, save_submission

train, test, sub = load_data()
X, y, X_test = make_xy(train, test)

# source 는 정답을 아는 train 전체. 개봉일이 더 이른 같은 감독 영화만 본다.
X = pd.concat([X, past_stats(train, train, 'director')], axis=1)
X_test = pd.concat([X_test, past_stats(test, train, 'director')], axis=1)
print('피처 {}개 -> {}개'.format(X.shape[1] - 4, X.shape[1]))

preds = []
for s in SEEDS:
    m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5,
                          random_seed=s, verbose=0, allow_writing_files=False)
    m.fit(X, np.log1p(y))
    preds.append(np.expm1(m.predict(X_test)))
pred = np.clip(np.mean(preds, axis=0), 0, None)

save_submission(pred, sub, 'submission_dirpast_raw.csv')

chk = test[['title']].copy()
chk['pred'] = pred.round().astype(int)
print()
print(chk[chk.title.isin(['명량', '베테랑', '군도: 민란의 시대', '사도', '신세계'])].to_string(index=False))
