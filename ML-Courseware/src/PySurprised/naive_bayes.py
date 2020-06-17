from sklearn.datasets import fetch_20newsgroups     # 导入新闻数据抓取器 fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # 导入文本特征向量化模块
from sklearn.naive_bayes import MultinomialNB       # 导入朴素贝叶斯模型
from sklearn.metrics import classification_report

# 1 数据获取
news = fetch_20newsgroups(subset='all')
print(len(news.data), len(news.target_names))

# 2 数据预处理，训练集和测试集分割，文本特征向量化
X_train, X_test, y_train, y_test = train_test_split(
    news.data, news.target, test_size=0.25, random_state=33
)                       # 随机采样 25% 的数据样本作为测试集
print(X_train[0])       # 查看训练样本
print(y_train[0:100])   # 查看标签

# 文本特征向量化
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 3 使用朴素贝叶斯进行训练
mnb = MultinomialNB()               # 使用默认配置初始化朴素贝叶斯
mnb.fit(X_train, y_train)           # 利用训练数据对模型参数进行估计
y_predict = mnb.predict(X_test)     # 对验证集进行预测

# 4 获取结果报告
print('The Accuracy of Naive Bayes Classifier is:', mnb.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=news.target_names).support)
