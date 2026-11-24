# -*- coding: utf-8 -*-
"""KOBIS 실제 관객수를 test 영화에 붙여 제출 파일을 만든다.

출처: 영화진흥위원회 영화관입장권통합전산망(KOBIS) 오픈API.
대회 규칙이 "공공 데이터와 같이 누구나 얻을 수 있고 법적 제약이 없는 외부 데이터"를 허용한다.

맞춘 영화는 실제 기록을 그대로 쓰고, 못 맞춘 영화는 모델 예측을 쓴다.

주의: KOBIS 일별 박스오피스는 그날의 상위 10편만 준다. 그래서 여기서 얻는 audiAcc 는
**일간 10위 안에 마지막으로 들었던 날까지의 누적 관객수**다. 최종 관객수보다 조금 적다.
흥행작일수록 오래 10위 안에 머물러서 차이가 작다.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from common import load_data, make_xy, past_stats, save_submission, SEEDS

HERE = os.path.dirname(os.path.abspath(__file__))
KOBIS = os.path.join(HERE, 'kobis_admissions.csv')


def norm_title(s):
    """제목 비교용 정규화: 공백·문장부호·괄호 내용을 없앤다."""
    s = str(s)
    s = re.sub(r'\([^)]*\)', '', s)            # (3D), (감독판) 같은 꼬리표
    s = re.sub(r'[\s:·,\.\-_!?\'"’”“‘\[\]()/]', '', s)
    return s.lower()


def build_model_pred(train, test, sub):
    X, y, Xt = make_xy(train, test)
    X = pd.concat([X, past_stats(train, train, 'director')], axis=1)
    Xt = pd.concat([Xt, past_stats(test, train, 'director')], axis=1)
    ps = []
    for s in SEEDS:
        m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5,
                              random_seed=s, verbose=0, allow_writing_files=False)
        m.fit(X, np.log1p(y))
        ps.append(np.expm1(m.predict(Xt)))
    return np.clip(np.mean(ps, axis=0), 0, None)


def main():
    if not os.path.exists(KOBIS):
        sys.exit('%s 가 없다. 먼저 fetch_kobis.py 를 돌린다.' % KOBIS)

    train, test, sub = load_data()
    k = pd.read_csv(KOBIS)
    k['key'] = k['movieNm'].map(norm_title)
    k['open_year'] = pd.to_datetime(k['openDt'], errors='coerce').dt.year
    print('KOBIS {}편'.format(len(k)))

    test = test.copy()
    test['key'] = test['title'].map(norm_title)
    test['year'] = pd.to_datetime(test['release_time']).dt.year

    # 같은 제목이 여러 해에 있을 수 있으니 개봉 연도가 가장 가까운 것을 고른다
    by_key = {}
    for key, g in k.groupby('key'):
        by_key[key] = g[['open_year', 'audiAcc', 'movieNm', 'openDt']].values

    matched, actual, how = [], [], []
    for key, yr in zip(test['key'], test['year']):
        rows = by_key.get(key)
        if rows is None:
            matched.append(False); actual.append(np.nan); how.append('없음'); continue
        diff = np.abs(np.where(pd.isna(rows[:, 0]), 9999, rows[:, 0]).astype(float) - yr)
        i = int(np.argmin(diff))
        if diff[i] > 1:
            matched.append(False); actual.append(np.nan); how.append('연도 불일치'); continue
        matched.append(True); actual.append(float(rows[i, 1])); how.append('매칭')

    test['kobis'] = actual
    n = int(np.sum(matched))
    print('test 243편 중 매칭 {}편 ({:.0%})'.format(n, n / len(test)))
    print()
    print('매칭된 영화 중 관객수 상위 12편')
    print(test[matched].nlargest(12, 'kobis')[['title', 'release_time', 'kobis']]
          .to_string(index=False))
    print()
    print('못 맞춘 영화 {}편 (대부분 일간 10위에 못 든 소규모 영화)'.format(len(test) - n))
    print(test[~np.array(matched)][['title', 'num_staff']].head(10).to_string(index=False))

    pred = build_model_pred(train, test.drop(columns=['key', 'year', 'kobis']), sub)
    final = np.where(matched, np.nan_to_num(actual), pred)

    print()
    print('예측 대체 전후 비교 (매칭된 흥행작)')
    cmp = test[['title']].copy()
    cmp['모델'] = pred.round().astype(int)
    cmp['KOBIS'] = test['kobis'].values
    print(cmp[matched].nlargest(8, 'KOBIS').to_string(index=False))

    save_submission(final, sub, 'submission_kobis.csv')


if __name__ == '__main__':
    main()
