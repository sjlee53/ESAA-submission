# -*- coding: utf-8 -*-
"""KOBIS(영화관입장권통합전산망) 오픈API로 2010~2015년 영화별 누적 관객수를 모은다.

왜 필요한가
-----------
대회 데이터의 `dir_prev_bfnum`(감독 이전작 평균 관객수)이 실제 기록과 맞지 않는다.
예) 베테랑은 97,469 로 적혀 있지만, 류승완 감독의 직전작 베를린은 716만 명이다.
KOBIS 실제 기록을 붙이면 감독·배급사 이력을 정확하게 만들 수 있다.
대회 규칙은 "공공 데이터와 같이 누구나 얻을 수 있고 법적 제약이 없는 외부 데이터"를 허용한다.

쓰는 법
-------
1. https://www.kobis.or.kr/kobisopenapi 에서 키를 발급받는다.
2. **키를 아래 KEY 에 직접 넣고 본인이 실행한다.**
   KOBIS 이용약관 4조에 따라 발급키는 타인에게 제공·공개·공유할 수 없다.
   채팅이나 깃허브에 올리지 말 것. (환경변수로 두는 쪽이 더 안전하다)

       set KOBIS_KEY=발급받은키        # 윈도우 cmd
       $env:KOBIS_KEY='발급받은키'     # 파워셸
       python fetch_kobis.py

3. 결과: kobis_admissions.csv  (movieCd, movieNm, openDt, audiAcc)
   이 파일만 공유하면 된다. 출처는 "영화진흥위원회 KOBIS"로 명시한다.

어떻게 모으나
-------------
일별 박스오피스는 그날의 상위 10편과 각 편의 **누적 관객수(audiAcc)**를 준다.
2010-01-01 ~ 2015-12-31 의 모든 날짜를 훑어 영화별 최대 audiAcc 를 취하면
그 기간에 한 번이라도 일간 10위 안에 든 영화의 최종 관객수를 얻는다.
오차를 좌우하는 흥행작은 전부 여기에 들어온다.

약 2,190회 호출이라 몇 분 걸린다. 중간에 끊겨도 다시 실행하면 이어서 받는다.
"""
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta

KEY = os.environ.get('KOBIS_KEY', '')      # 여기에 직접 적어도 된다 (공유 금지)


def key_from_file(path):
    """KOBIS 에서 내려받은 발급키 목록 파일에서 키만 꺼낸다 (화면에 찍지 않는다).

    키는 32자리 16진수다. 파일이 진짜 엑셀이든 HTML 표든 상관없이 찾아낸다.
    """
    import re
    raw = open(path, 'rb').read()
    for enc in ('utf-8', 'cp949', 'utf-16', 'latin-1'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode('latin-1', 'ignore')
    found = re.findall(r'\b[0-9a-f]{32}\b', text, re.I)
    if not found:
        raise SystemExit('그 파일에서 32자리 발급키를 찾지 못했다: %s' % path)
    return found[0]


if not KEY and len(sys.argv) > 1:
    KEY = key_from_file(sys.argv[1])
    print('발급키를 파일에서 읽었다 (끝 4자리 ****%s)' % KEY[-4:])

START = date(2010, 1, 1)
END = date(2015, 12, 31)
URL = ('http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/'
       'searchDailyBoxOfficeList.json')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kobis_admissions.csv')
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_kobis_cache.json')
WORKERS = 8                                 # 동시 요청 수 (순차로는 77분 걸린다)


def fetch_day(day):
    q = urllib.parse.urlencode({'key': KEY, 'targetDt': day.strftime('%Y%m%d')})
    with urllib.request.urlopen(URL + '?' + q, timeout=30) as r:
        d = json.loads(r.read().decode('utf-8'))
    if 'faultInfo' in d:
        raise SystemExit('KOBIS 오류: %s (키를 확인하세요)' % d['faultInfo'].get('message'))
    return d['boxOfficeResult']['dailyBoxOfficeList']


def main():
    if not KEY:
        print('KOBIS_KEY 가 비어 있다. 환경변수로 넣거나 발급키 목록 파일을 인자로 준다.')
        sys.exit(1)

    movies, done = {}, set()
    if os.path.exists(CACHE):
        saved = json.load(io.open(CACHE, encoding='utf-8'))
        movies, done = saved['movies'], set(saved['done'])
        print('이어서 받는다: 이미 %d일 완료, 영화 %d편' % (len(done), len(movies)))

    days, d = [], START
    while d <= END:
        if d.strftime('%Y%m%d') not in done:
            days.append(d)
        d += timedelta(days=1)
    print('남은 일수 %d일, 동시 요청 %d개' % (len(days), WORKERS))

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()

    def work(day):
        for attempt in range(3):
            try:
                return day, fetch_day(day)
            except SystemExit:
                raise
            except Exception:
                time.sleep(1.0 + attempt)
        return day, None

    n = 0
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for fut in as_completed([ex.submit(work, d) for d in days]):
                day, rows = fut.result()
                n += 1
                if rows is None:
                    continue
                with lock:
                    for m in rows:
                        cd, acc = m['movieCd'], int(m['audiAcc'])
                        cur = movies.get(cd)
                        if cur is None or acc > cur['audiAcc']:
                            movies[cd] = {'movieCd': cd, 'movieNm': m['movieNm'],
                                          'openDt': m.get('openDt', ''), 'audiAcc': acc}
                    done.add(day.strftime('%Y%m%d'))
                    if n % 200 == 0:
                        print('  %d/%d일, 영화 %d편' % (n, len(days), len(movies)), flush=True)
                        json.dump({'movies': movies, 'done': sorted(done)},
                                  io.open(CACHE, 'w', encoding='utf-8'), ensure_ascii=False)
    except KeyboardInterrupt:
        print('중단됨. 지금까지 받은 것을 저장한다.')
    finally:
        json.dump({'movies': movies, 'done': sorted(done)},
                  io.open(CACHE, 'w', encoding='utf-8'), ensure_ascii=False)

    import pandas as pd
    df = pd.DataFrame(movies.values()).sort_values('audiAcc', ascending=False)
    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    print()
    print('저장: %s  (%d편)' % (OUT, len(df)))
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
