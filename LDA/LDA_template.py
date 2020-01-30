import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 임의의 X, y값 지정.
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

# LDA모델 구축.
clf = LinearDiscriminantAnalysis()
clf.fit(X,y)

# predict.
print(clf.predict([[-0.8, -1]]))

# QDA를 이용한 예측.
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

clf2 = QuadraticDiscriminantAnalysis()
clf2.fit(X,y)

# confusion matrix를 통해 LDA와 QDA 비교.
from sklearn.metrics import confusion_matrix
y_pred=clf.predict(X)
confusion_matrix(y,y_pred)  
y_pred2=clf2.predict(X)
confusion_matrix(y,y_pred2)  
