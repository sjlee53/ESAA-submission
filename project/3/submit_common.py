# -*- coding: utf-8 -*-
"""common.py 기준 구성 그대로의 제출 파일 (감독 과거작 피처 없음).

첫 제출(baseline_catboost.py)과 다른 점은 배급사 표기 정규화와 time 보정이 들어간 것이다.
이 점수를 알아야 감독 과거작 피처의 효과를 따로 떼어 볼 수 있다.
"""
import numpy as np
from catboost import CatBoostRegressor
from common import load_data, make_xy, fit_predict, save_submission

train, test, sub = load_data()
X, y, X_test = make_xy(train, test)
model = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5,
                          verbose=0, allow_writing_files=False)
pred = fit_predict(model, X, y, X_test)
save_submission(pred, sub, 'submission_common.csv')

chk = test[['title']].copy(); chk['pred'] = pred.round().astype(int)
print()
print(chk[chk.title.isin(['명량', '베테랑', '군도: 민란의 시대', '사도', '신세계'])].to_string(index=False))
