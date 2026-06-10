

import sys

from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle
from PyQt5.QtCore import QStringListModel

# from job04_recommendation import sentence, recommendation
# from job04_recommendation import cosine_sim, recommendation

form_window = uic.loadUiType('./movie_recommendation.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # tfidf 매트릭스 가져오기
        self.Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr() # 벡터연산 최적화
        with open('./models/tfidf.pkl', 'rb') as f:
            self.Tfidf = pickle.load(f)
        # embadding모델 가져오기
        self.embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')


        self.df_reviews = pd.read_csv('./datasets/reviews_2017_2022.csv') # 판다스로 선언해서 다얗한 함수를 쓰기 위함
        self.titles = list(self.df_reviews.titles)
        self.titles.sort()
        for title in self.titles:
            self.cb_title.addItem(title)

        # 자동완성 기능 추가
        # 컴플리터를 만들고 자동완성을 위한 모델만들기
        # model = QStandardItemModel()
        model = QStringListModel()
        model.setStringList((self.titles))
        completer =  QCompleter()
        completer.setModel(model)
        self.le_keyword.setCompleter(completer)

        #combobox
        # qt designer에 curruntIndexChanged를 edit signal에서 볼수 있다.
        self.cb_title.currentIndexChanged.connect(self.combobox_slot) # 함수 이름을 준다. ()안됨 [실행X]
        # 추천 버튼 signal도착시 콜백함수 연결
        self.btn_recommend.clicked.connect(self.btn_keywords_clicked)

    def btn_keywords_clicked(self):
        keyword = self.le_keyword.text()
        # 자동완성 제목으로 선택하면 1번방법으로 추천해주고 아니면 2번 방법으로 추천해준다.
        if keyword in self.titles:
            recommendations = self.recommendation_by_title(keyword)
        else:
            recommendations = self.recommendation_by_keyword(keyword)
        self.lb_recommendation.setText(recommendations)


    def getRecommendation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))  # 인덱스 만들기 [0]에 들어감
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)  # 정렬
        simScore = simScore[:11]
        moviIdx = [i[0] for i in simScore]  # 찾은 녀석의 인덱스 반환
        recmovieList = self.df_reviews.iloc[moviIdx, 0]  # 0번이 제목임 제목만 뽑아서 리턴
        return recmovieList[:11]  # 자기자신이 가장 정확도가 높기 때문에 본인은 제외

    # 1. 좋아하는 영화를 기반으로 연관도 높은 영화 추천
    # combobox누르면 signal이 발생한다.
    def combobox_slot(self):
        title = self.cb_title.currentText() # 데이터 선택
        print(title) # 터미널에 콤보박스 선택하면 실행되는것 확인 가능
        recommendation = self.recommendation_by_title(title) # 연관성높은 10개 가져온다.
        self.lb_recommendation.setText(str(recommendation))

    def recommendation_by_title(self, title):
        movieIdx = self.df_reviews[self.df_reviews['titles'] == title].index[0] # 영화의 인덱스만 빼내기
        print(movieIdx)
        cosine_sim = linear_kernel(self.Tfidf_matrix[movieIdx], self.Tfidf_matrix)
        recommendation = self.getRecommendation(cosine_sim)
        #라벨에 출력 문자열 + 줄바꿈
        recommendation = '\n'.join(recommendation[1:])
        return recommendation

    # 2. 키워드 추천 받아서 연관 되어있는 영화 추천하기
    def recommendation_by_keyword(self, keyword):
        try:
            sim_word = self.embedding_model.wv.most_similar(keyword, topn=10)
        except:
            return "제가 모르는 단어에요"
        sentence = [keyword] * 11
        count = 10
        for word, _ in sim_word:
            sentence = sentence + [word] * count
            count = count - 1
        print(sentence)
        sentence = ' '.join(sentence)
        print(sentence)
        sentence_vec = self.Tfidf.transform([sentence])
        cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrix)
        recommendations = self.getRecommendation(cosine_sim)
        recommendations = '\n'.join(recommendations[:10])
        return recommendations




if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())


