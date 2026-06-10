
# 1 / text fre(빈도수) = tfidf의 벡터값은 전체에 공통으로 많이 나오는녀석은 작고 적게나오면 높다.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle

df_reviews = pd.read_csv('./datasets/reviews_2017_2022.csv')
df_reviews.info()

Tfidf = TfidfVectorizer(sublinear_tf=True)
Tfidf_matrix = Tfidf.fit_transform(df_reviews.reviews)
print(Tfidf_matrix.shape)
# 3174개의 문장
# 유니크한 형태소의 갯수 83709개 에 대한 TF를 가진다.

# 매트릭스 저장
with open('./models/tfidf.pkl', 'wb') as f:
    pickle.dump(Tfidf, f)

# 매트릭스를 저장할때 mmwrite사용한다.
mmwrite('./models/Tfidf_movie_review.mtx', Tfidf_matrix)

# ftidf만으로 영화추전이 가능
