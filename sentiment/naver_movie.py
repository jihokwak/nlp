import requests
from bs4 import BeautifulSoup
from datetime import datetime as dt
import pandas as pd

#크롤링
review_data = pd.DataFrame(columns=["review_cd", "score", "review", "user", "write_date"])

for page_nm in range(1, 1001) :
    resp  = requests.get("https://movie.naver.com/movie/point/af/list.nhn?st=mcode&sword=132623&target=after&page={}".format(page_nm))
    soup = BeautifulSoup(resp.text,"html.parser")

    review_cds = [cd.text.strip() for cd in  soup.select("td.ac")]
    scores = [int(score.text.strip()) for score in  soup.select("td.point")]
    reviews  = [review.text.strip().split("\n")[1] for review in  soup.select("td.title")]
    users = [user.text.strip() for user in  soup.select("td.num > a")]
    write_dates = [date.text.strip()[-8:] for date in  soup.select("td.num")][1:][::2]
    write_dates = [dt.date(dt.strptime(date, '%y.%m.%d')) for date in write_dates]

    cur_page_df = pd.DataFrame({'review_cd':review_cds,
                  'score':scores,
                  'review':reviews,
                  'user':users,
                  'write_date':write_dates})
    print(str(page_nm) + "페이지 크롤링")
    print(cur_page_df)
    review_data = pd.concat([review_data, cur_page_df], axis=0).reset_index(drop=True)

#불용어 제거
import re
review_data.review = review_data.review.apply(lambda x : re.sub(r'[\W]+', ' ', x))

#데이터 저장
import os
CUR_DIR = os.path.abspath(".")
PJT_DIR = "sentiment"
DATA_DIR = "data"
os.makedirs(os.path.join(CUR_DIR, PJT_DIR, DATA_DIR), exist_ok=True)

review_data.to_csv(os.path.join(CUR_DIR, PJT_DIR, DATA_DIR, "naver_movie_review_data.csv"))

#WPM 학습
from subword_nmt.learn_bpe import learn_bpe
import io
with open(os.path.join(CUR_DIR, PJT_DIR, DATA_DIR, "영화평BPE.txt"), 'w', encoding='utf8') as outfile:
    infile = io.StringIO(' '.join(review_data.review.tolist()))
    learn_bpe(infile, outfile, 1000)

from subword_nmt.apply_bpe import BPE
with open(os.path.join(CUR_DIR, PJT_DIR, DATA_DIR, "영화평BPE.txt"), encoding='utf8') as f:
    bpe = BPE(f, separator='~')

#TDM 만들기

def tokenizer_wpm(text):
    tokens = bpe.process_line(text).split()
    return [t for t in tokens
            if (not t.endswith('~') and len(t) > 1) or len(t) > 2]

from sklearn.feature_extraction.text import CountVectorizer
cv_wpm = CountVectorizer(max_features=1000, tokenizer=tokenizer_wpm)
tdm = cv_wpm.fit_transform(review_data.review)

#토큰 빈도
freq = pd.DataFrame({
    'word': cv_wpm.get_feature_names(),
    'n': tdm.sum(axis=0).flat
})

#데이터분할
import numpy as np
review_data['sentiment'] = np.where(review_data.score > 5, 1, 0)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    tdm, review_data.sentiment, test_size=.2, random_state=1234)


#학습
from sklearn.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(random_state=1234)
model.fit(x_train, y_train)

#성능평가
from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)

#계수분석
word_coef = pd.DataFrame({
    'word': cv_wpm.get_feature_names(),
    'coef': model.coef_.flat
})
word_coef.sort_values('coef').head(10)
word_coef.sort_values('coef').tail(10)