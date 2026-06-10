# word_to_vec : 단어를 의미공간에 벡터화
# 의미학습을 하는법은
# 의미벡터를 만들고 의미연산이 된다. LSTM.(RNN)

import pandas as pd
from gensim.models import Word2Vec

df_reviews = pd.read_csv('datasets/reviews_2017_2022.csv')
df_reviews.info()

reviews = list(df_reviews.reviews)
print(reviews[0])


tokens = []
for sentence in reviews:
    token = sentence.split()
    tokens.append(token)
print(tokens[0]) # split()이니까 띄어쓰기를 기준으로 나눔
#okt는 필요없다 이미 csv에서 되어있다.

embedding_model = Word2Vec(tokens,vector_size=100,window=4,min_count=20,workers=16, epochs=100, sg=1)
# 토클을 주면 안에 있는 형태소의 갯수차원을 만들고 벡터화를 한다.
# 차원이 커질수록 데이터가 희소해진다.
# 차원이 많을수록 거리가 멀어진다. -> 공간상에 데이터가 희소해진다.
# 차원이 늘어남에 따라 데이터는 제곱만큼 늘어나야 한다. x^2 + y^2 ==> z^2 + t^2

# 데이터 차원을 축소한다. sg=1인 알고리즘을 사용한다.
# 차원축소는 어떻게 할까?
# 3차원 공간을 이미지로 찍으면 2차원이 된다.
# 주사위가 있는데 사진을 찍는 위치에 따라서 정보가1,2,3개일 수 있다.
# 정보가 3개가 보이게 주사위의 입체적인 면이 최대한 많이 보이게 찍는방식으로 데이터손실을 최대한 줄인다.

# 2차원에 y = x를 따라서 점들이 있다고 하면
# y=x인 축을 차원으로 하면 거리가 y축을 버리면 x차원만 남지만 점들간의 거리가 감소한다.
# 그러나 y = -x에서 투영하면 거리에 손실이 나지 않는다.

# window
# 여러개의 단어들을 한번에 하지않고 10개가 있으면 4개씩 잘라서 컨볼루션 필터처럼 본다.
#

# min_count : 20번이상 등장하는 녀석들만 의미벡터를 만든다.

# 근처에 모여있으면 유사하다고 판단한다.
embedding_model.save('./models/word2vec_movie_review.model')
print(list(embedding_model.wv.index_to_key))
print(len(embedding_model.wv.index_to_key))
# 엄청나게 오래걸린다.


