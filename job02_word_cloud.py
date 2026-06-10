
import pandas as pd
from wordcloud import WordCloud
import collections
import matplotlib.pyplot as plt
from matplotlib import font_manager,rc

# wordcloud :

font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font', family='NamumBarunGothis')

df = pd.read_csv('./datasets/reviews_2017_2022.csv')

# 16,308
# 22,169
# 300, 1228
movie_index1 = 300
movie_index2 = 1228
words1 = df.iloc[movie_index1,1].split() # n번쨰 영화를 보고 추천받기.
print(df.iloc[movie_index1,0])
words2 = df.iloc[movie_index2,1].split() # n번쨰 영화를 보고 추천받기.
print(df.iloc[movie_index2,0])

worddict1 = collections.Counter(words1)
wordlist1 = dict(worddict1)
worddict2 = collections.Counter(words2)
wordlist2 = dict(worddict2)
print(wordlist1,wordlist2)

wordcloud1 = WordCloud(font_path=font_path,background_color='white',stopwords=wordlist1).generate_from_frequencies(wordlist1)
wordcloud2 = WordCloud(font_path=font_path,background_color='white',stopwords=wordlist2).generate_from_frequencies(wordlist2)
# plt.imshow(wordcloud1, wordcloud2)
# plt.axis("off")
# plt.show() #pyqt5설치 필요

fig, axes = plt.subplots(1,2, figsize=(10,5))
axes[0].imshow(wordcloud1)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[1].imshow(wordcloud2)
axes[1].set_xticks([])
axes[1].set_yticks([])
plt.show()

# 공통된 부분만 보고 싶다.