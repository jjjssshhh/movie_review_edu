

# 전처리

import pandas as pd
from konlpy.tag import Okt
import re

df = pd.read_csv('./datasets/reviews_2017_2022.csv')
df.info()

df_stopwords = pd.read_csv('./datasets/stopwords.csv')
stopwords = df_stopwords['stopword'].tolist()                   # 제거할 불용어 모음집
                                                                # 특별히 뺴줄 녀석 추가선택
stopwords = stopwords + ['가다','감독','연출','연기','배우','하다','모르다','보여주다','주연','많다','좋다']

okt = Okt()
print(df.titles[0])
print(df.reviews[0])
tokened_review = okt.pos(df.reviews[0]) # stem처럼 단어로 쪼개기 +
# pos는 품사로 태깅까지 해준다. 명사,동사 등
print(tokened_review)

cleaned_sentences = []
for review in df.reviews[:5]:
    review = re.sub('[^가-힣]', ' ', review) # 영문 특수문자 제외
    tokened_review = okt.pos(review, stem=True) #
    df_token = pd.DataFrame(tokened_review, columns=['word', 'class'])# 형태소,품사
    df_token = df_token[(df_token['class'] == 'Noun') |
                        (df_token['class'] == 'Verb') |
                        (df_token['class'] == 'Adjective')] # 명사,동사,부사만 넣어서 토큰 만들기
    words = []
    for word in df_token['word']:
        if len(word) > 1:
            if word not in stopwords: # 지정한 녀석도 제외
                words.append(word)

    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)

# df.reviews[:5] = cleaned_sentences
df.loc[:4, 'reviews'] = cleaned_sentences

df.dropna(inplace=True)
df.info()

df.to_csv('./datasets/reviews_2017_2022_test.csv', index=False) # 인덱스 미포함


