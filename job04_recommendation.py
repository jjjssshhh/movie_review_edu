import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
from gensim.models import Word2Vec

# from job05_word2vec import sentence
# from job03_TFIDF import df_reviews


def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1])) # 인덱스 만들기 [0]에 들어감
    simScore = sorted(simScore, key=lambda x: x[1],reverse=True) # 정렬
    simScore = simScore[:11]
    moviIdx = [i[0] for i in simScore]  # 찾은 녀석의 인덱스 반환
    recmovieList = df_reviews.iloc[moviIdx, 0] # 0번이 제목임 제목만 뽑아서 리턴
    return recmovieList[:11] # 자기자신이 가장 정확도가 높기 때문에 본인은 제외

df_reviews = pd.read_csv('./datasets/reviews_2017_2022.csv')
Tfidf_matrix = mmread('models/Tfidf_movie_review.mtx').tocsr()

with open('./models/tfidf.pkl', 'rb') as f:
    Tfidf = pickle.load(f) # Tfidx객체 가져오기

# # =================================
# # 1. 영화 index 이용
# # list - 2
# ref_idx = 1228
# print('title',df_reviews.iloc[ref_idx,0])
# cosine_sim = linear_kernel(Tfidf_matrix[ref_idx], Tfidf_matrix)
# # cosine_sim
# # 영화 벡터의좌표가 많은데.
# # 벡터가 되면 두 개의 값사이의 cosine값을 구할 수 있다.
# # 즉 같은 방향을 보고있을 수록 높은값을 가지게된다.
# # 크기가 같지 않아도 된다. '방향'만 같아도 된다.
# # 차원이 많아질수록 0에 가까워진다.
# # tfidx의 하나와 나머지를 각각 cosine값을 출력한다.
#
# print(cosine_sim[0])
# print(len(cosine_sim))
#
# recommendations = getRecommendation(cosine_sim)
# print(recommendations[1:11])
# # =================================

# =================================
# 2. 영화 index 이용
# word_to_vec : 단어를 의미공간에 벡터화
# 의미학습을 하는법은
# 의미벡터를 만들고 의미연산이 된다. LSTM.(RNN)
embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')
keyword = '대왕' # 한글 + 두글자 이상

if keyword not in list(embedding_model.wv.index_to_key):
    print('단어에 없습니다.\n')
else:
    sim_word = embedding_model.wv.most_similar(keyword,topn=10)
    print(sim_word)
    # 키워드와 가장유사한 단어를 10개 출력한다.
    sentence = [keyword] * 11
    count = 10
    for word,_ in sim_word:
        sentence = sentence + [word] * count
        count = count -1

    print(sentence)
    sentence = ' '.join(sentence)
    print(sentence)

    sentence_vec = Tfidf.transform([sentence])
    cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
    recommendation = getRecommendation(cosine_sim)
    print(recommendation)
    # 키워드에 없는 단어를 고르면 에러가 뜬다.




















