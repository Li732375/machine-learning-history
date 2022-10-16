###機器學習教材
### https://www.datacamp.com/community/tutorials/scikit-learn-python


import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.manifold import Isomap
from sklearn import metrics
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn import cluster
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import joblib



# Load in the `digits` data
digits = datasets.load_digits()

print(digits)

##顯示所有可指定的欄位名稱
print('\nkeys >>')
print(digits.keys())
#print(digits.columns)

#顯示指定欄的數值陣列
print('\ndata >>')
print(digits.data)

#顯示指定欄的數值陣列
print('\ntarget >>')
print(digits.target)

#指定欄的數值陣列
print('\nDESCR >>')
print(digits.DESCR)

##顯示該屬性(欄)各維度的大小

digits_data = digits.data

# data 的各維度大小
print('\ndigits_data.shape >>')
print(digits_data.shape)

# Isolate the target values with `target`
digits_target = digits.target

# target 的各維度大小
print('\ndigits_target.shape >>')
print(digits_target.shape)

# numpy.unique(陣列)會分析出陣列裡的所有數值的集合
#顯示數值區間
#數值種類個數
number_digits = len(np.unique(digits.target))
print(np.unique(digits.target))
print(number_digits)

# Isolate the `images`
digits_images = digits.images

# Inspect the shape
# images 的各維度大小
print('\ndigits_images.shape >>')
print(digits_images.shape)

'''
# 載入 matplotlib
import matplotlib.pyplot as plt

# 載入 `digits`
#digits = datasets.load_digits()

# 設定圖形的大小（寬, 高）
fig = plt.figure(figsize=(4, 2))

# 調整子圖形 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# 把前 8 個手寫數字顯示在子圖形
for i in range(8):
    # 在 2 x 4 網格中第 i + 1 個位置繪製子圖形，並且關掉座標軸刻度
    ax = fig.add_subplot(2, 4, i + 1, xticks = [], yticks = [])
    # 顯示圖形，色彩選擇灰階
    ax.imshow(digits.images[i], cmap = plt.cm.binary)
    # 在左下角標示目標值
    ax.text(0, 7, str(digits.target[i]))

# 顯示圖形
plt.show()
'''

##視覺化繪圖


# 將觀測值與目標值放入一個 list
# zip(a, b) 打包 a, b 成數組
images_and_labels = list(zip(digits.images, digits.target))

# list 中的每個元素
# enumerate(陣列) 遍歷陣列，倘若為一維，則搭配索引值
'''
# ex: enumerate(陣列, 1)，表示搭配的索引值從 1 (和陣列索引無關)開始

my_list = ['apple', 'banana', 'grapes', 'pear']
for c, value in enumerate(my_list, 1):
    print(c, value)
>>True

for c, value in enumerate(1, my_list):
    print(c, value)
>>TypeError: 'list' object cannot be interpreted as an integer

for c, value in enumerate(my_list, my_list):
    print(c, value)
>>TypeError: 'list' object cannot be interpreted as an integer

for c, value in enumerate(list(zip(my_list, my_list))):
    print(c, value)
>>True
'''
#下面因每次都要關圖片才能繼續，所以先 ban 起來
'''
for i, (image, label) in enumerate(images_and_labels[:10]):
    
    # 在 i + 1 的位置初始化子圖形，疑似 matplotlib 是從 1 開始
    # plt.subplots(高, 寬, 索引值)在一個視窗繪製多個圖表
    plt.subplot(3, 4, i + 1)
    
    # 關掉子圖形座標軸刻度
    plt.axis('off')
    
    # 顯示圖形，色彩選擇灰階
    plt.imshow(image, cmap = plt.cm.binary)
    
    # 加入子圖形的標題
    plt.title('Training: ' + str(label))

# 顯示圖形
plt.show()
'''

##因 digits 資料有 64 個變數，需要降維處理（Dimensionality Reduction）
##採用其中的方法，主成份分析（Principal Component Analysis, PCA）
##找出變數之間的線性關係組成新的一個主成份，然後使用這個
##主成份取代原有的變數，屬於一種最大化資料變異性的線性轉換方法



'''
##因下列程式碼已過期，如全壘打般回不來了~

# Create a Randomized PCA model that takes two components
randomized_pca = RandomizedPCA(n_components = 2)

# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(digits.data)

# Inspect the shape
print("Shape of reduced_data_pca:", reduced_data_pca.shape)
print("---")

print("RPCA:")
print(reduced_data_rpca)
print("---")

'''

# Create a regular PCA model 
pca = PCA(n_components = 2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(digits.data)

# Print out the data
print("PCA:")
print(reduced_data_pca)


##將結果視覺化
##說是可以增加對資料的認知，雖然我也覺得沒有比較好的樣子就是了~

#下面因每次都要關圖片才能繼續，所以先 ban 起來
'''
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    x = reduced_data_pca[:, 0][digits.target == i]
    y = reduced_data_pca[:, 1][digits.target == i]
    plt.scatter(x, y, c=colors[i])
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()
'''

##準備對 digits 資料使用非監督式學習演算法
##資料的標準化



# Apply `scale()` to the `digits` data
#將這 64 個維度的分佈轉換為平均數為 0，標準差為 1 的標準常態分佈。
data = scale(digits.data)

print(data)


##切分資料為訓練與測試資料
##常見的切分比例是 2/3 作為訓練資料，1/3 作為測試資料。

# Import `train_test_split`
#下行的模組已走入歷史，不能用
#from sklearn.cross_validation import train_test_split


# Split the `digits` data into training and test sets
# test_size  可以設定 train_size 或 test_size，只要設定一邊即可，範圍在 [0-1] 之間
#這裡設定 1/4 作測試資料比例
# random_state  為亂數種子，可以固定我們切割資料的結果
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data,
                                                                               digits.target,
                                                                               digits.images,
                                                                               test_size = 0.25,
                                                                               random_state = 42)




##檢視訓練資料的觀測值與目標值資訊
##訓練完了？(懷疑人生中~)
##其實還沒訓練啦 ~ 訓練前的資料檢視而已

# Number of training features
n_samples, n_features = X_train.shape

# Print out `n_samples`
print(n_samples)

# Print out `n_features`
print(n_features)

# Number of Training labels
n_digits = len(np.unique(y_train))

# Inspect `y_train`
print(len(y_train))



##應用 K-Means 演算法
##開始訓練



# Create the KMeans model
#init 參數指定我們使用的 K-Means 演算法 k-means++，預設就是使用 K-Means 演算法，參數其實可以省略

#n_clusters 參數被設定為 10，這呼應了我們有 0 到 9 這 10 個相異目標值。

#假使在未知群集的情況下，通常會嘗試幾個不同的 n_clusters 參數值，分別計算平方誤差和（Sum of the
# Squared Errors, SSE），然後選擇平方誤差和最小的那個 n_clusters 作為群集數
clf = cluster.KMeans(init = 'k-means++', n_clusters = 10, random_state = 42)

# Fit the training data to the model
clf.fit(X_train)



##視覺化訓練結果

#下面因每次都要關圖片才能繼續，所以先 ban 起來
'''
# 設定圖形的大小
fig = plt.figure(figsize=(8, 3))

# 圖形標題
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# 對所有的目標值（0 - 9）
for i in range(10):
    # 在 2x5 的網格上繪製子圖形
    ax = fig.add_subplot(2, 5, i + 1)
    # 顯示圖片
    ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    # 將座標軸刻度關掉
    plt.axis('off')

# 顯示圖形
plt.show()
'''



##預測測試資料的目標值

# Predict the labels for `X_test`
y_pred=clf.predict(X_test)

# Print out the first 100 instances of `y_pred`
print(y_pred[:100])

# Print out the first 100 instances of `y_test`
print(y_test[:100])

# Study the shape of the cluster centers
clf.cluster_centers_.shape



##將預測的目標值視覺化
##並加入主成份分析重新執行程式，觀察跟 Isomap 有什麼差異



#下面因每次都要關圖片才能繼續，所以先 ban 起來
'''
# 使用 Isomap 對 `X_train` 資料降維
X_iso = Isomap(n_neighbors = 10).fit_transform(X_train)

# 使用 K-Means 演算法
K_Means_clusters = clf.fit_predict(X_train)

# 在 1x2 的網格上繪製子圖形
fig, ax = plt.subplots(2, 2, figsize = (8, 7))

# 調整圖形的外觀
fig.suptitle('Predicted Versus Training Labels', fontsize = 14, fontweight = 'bold')
fig.subplots_adjust(top = 0.85)

# 加入散佈圖 
ax[0][0].scatter(X_iso[:, 0], X_iso[:, 1], c = K_Means_clusters)
ax[0][0].set_title('K-Means_Predicted Training Labels')
ax[0][1].scatter(X_iso[:, 0], X_iso[:, 1], c = y_train)
ax[0][1].set_title('K-Means_Actual Training Labels')

# 使用 PCA 對 `X_train` 資料降維
X_pca = PCA(n_components = 2).fit_transform(X_train)

# 使用 PCA 演算法
PCA_clusters = clf.fit_predict(X_train)

# 加入散佈圖 
ax[1][0].scatter(X_pca[:, 0], X_pca[:, 1], c = PCA_clusters)
ax[1][0].set_title('PCA_Predicted Training Labels')
ax[1][1].scatter(X_pca[:, 0], X_pca[:, 1], c = y_train)
ax[1][1].set_title('PCA_Actual Training Labels')

# 顯示圖形
plt.show()
'''


##評估分群模型的表現
##印出混淆矩陣（Confusion matrix）



# Print out the confusion matrix with `confusion_matrix()`
print(metrics.confusion_matrix(y_test, y_pred))

##換個方式呈現

print('\nclf.inertia_: %i' % clf.inertia_)
print('inertia: %.3f' % (homogeneity_score(y_test, y_pred)))

#Completeness score 則告訴我們一定有觀測值被分在錯誤的群集
print('homo: %.3f' % (completeness_score(y_test, y_pred)))
print('compl: %.3f' % (v_measure_score(y_test, y_pred)))
print('v-meas: %.3f' % (adjusted_rand_score(y_test, y_pred)))

#ARI 則告訴我們同一群集中的觀測值沒有完全相同
print('ARI AMI: %.3f' % (adjusted_mutual_info_score(y_test, y_pred)))

#silhouette score 接近 0，代表很多的觀測值都接近分群邊界而可能被分到錯誤的群集中
print('silhouette: %.3f' % (silhouette_score(X_test, y_pred, metric = 'euclidean')))



##嘗試另外一種演算法：支持向量機（Support Vector Machines, SVM）

#跟先前同樣的比例切出測試資料

##開始 SVM 訓練 1
# kernel = 'linear'，設為 S1
# Split the data into training and test sets 
#前面 267 行有切過，不用再切



# Create the SVC model 
svc_model = svm.SVC(gamma = 0.001, C = 100., kernel = 'linear')

# Fit the data to the SVC model
svc_model.fit(X_train, y_train)




##開始 SVM 訓練 2
##使用網格搜索（Grid search）自動找出合適的參數設定，設為 S2

# Split the `digits` data into two equal sets
S2_X_train, S2_X_test, S2_y_train, S2_y_test = train_test_split(digits.data, digits.target,
                                                                test_size = 0.5, random_state = 0)

# Import GridSearchCV
#下行的模組已走入歷史，不能用
#from sklearn.grid_search import GridSearchCV


# Set the parameter candidates
#使用網格搜索（Grid search）
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},]

# Create a classifier with the parameter candidates
GridSearchCV_clf = GridSearchCV(estimator = svm.SVC(), param_grid = parameter_candidates, n_jobs = -1)

# Train the classifier on training data
GridSearchCV_clf.fit(S2_X_train, S2_y_train)

# Print out the results 
print('Best score for training data:', GridSearchCV_clf.best_score_)
print('Best `C`:',GridSearchCV_clf.best_estimator_.C)
print('Best kernel:',GridSearchCV_clf.best_estimator_.kernel)
print('Best `gamma`:',GridSearchCV_clf.best_estimator_.gamma)



##比較手動設定參數與使用網格搜索調整參數的兩個分類器，檢視網格搜索找出來的參數是否真的比較好

# Apply the classifier to the test data, and view the accuracy score
GridSearchCV_clf.score(S2_X_test, S2_y_test)

# Train and score a new classifier with the grid search parameters
print('GridSearchCV\'s parameter, accuracy score: ', end = '')
print(svm.SVC(C = 10, kernel = 'rbf', gamma = 0.001).fit(S2_X_train, S2_y_train).score(S2_X_test,S2_y_test))



##進行預測測試
# Predict the label of `X_test`
print(svc_model.predict(X_test))

# Print `y_test` to check the results
#答案卷
print(y_test)



##視覺化結果


# 將預測結果指派給 `predicted`
predicted = svc_model.predict(X_test)

# 將 `images_test` 與 `predicted` 存入 `images_and_predictions`
images_and_predictions = list(zip(images_test, predicted))

#下面因每次都要關圖片才能繼續，所以先 ban 起來
'''
# 繪製前四個元素
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    # 在 1x4 的網格上繪製子圖形
    plt.subplot(1, 4, index + 1)
    # 關掉座標軸的刻度
    plt.axis('off')
    # 色彩用灰階
    plt.imshow(image, cmap = plt.cm.binary)
    # 加入標題
    plt.title('Predicted: ' + str(prediction))

# 顯示圖形
plt.show()
'''



##評估分群模型的表現
##印出混淆矩陣（Confusion matrix）



# Print the classification report of `y_test` and `predicted`
print(metrics.classification_report(y_test, predicted))

# Print the confusion matrix
print(metrics.confusion_matrix(y_test, predicted))



##視覺化預測結果與目標值
##將前述的兩種方式也一併進來比較

svc_model = svm.SVC(gamma = 0.001, C = 100., kernel = 'linear')
svc_model.fit(X_train, y_train)

# 對 `digits` 資料降維
X_iso = Isomap(n_neighbors = 10).fit_transform(X_train)

# 使用 SVC 演算法
predicted = svc_model.predict(X_train)

# 使用 K-Means 演算法
K_Means_clusters = clf.fit_predict(X_train)

# 使用 PCA 對 `X_train` 資料降維
X_pca = PCA(n_components = 2).fit_transform(X_train)

# 使用 PCA 演算法
PCA_clusters = clf.fit_predict(X_train)

#下面因每次都要關圖片才能繼續，所以先 ban 起來
'''
# 在 1x2 的網格上繪製子圖形
fig, ax = plt.subplots(2, 3, figsize = (11, 6.5))

# 調整圖形的外觀
fig.suptitle('Predicted Versus Training Labels', fontsize = 12, fontweight = 'bold')
fig.subplots_adjust(wspace = 0.2, top = 0.9)

# 加入 K-Means 散佈圖 
ax[0][0].scatter(X_iso[:, 0], X_iso[:, 1], c = K_Means_clusters)
ax[0][0].set_title('K-Means_Predicted Training Labels')
ax[1][0].scatter(X_iso[:, 0], X_iso[:, 1], c = y_train)
ax[1][0].set_title('K-Means_Actual Training Labels')

# 加入 PCA 散佈圖 
ax[0][1].scatter(X_pca[:, 0], X_pca[:, 1], c = PCA_clusters)
ax[0][1].set_title('PCA_Predicted Training Labels')
ax[1][1].scatter(X_pca[:, 0], X_pca[:, 1], c = y_train)
ax[1][1].set_title('PCA_Actual Training Labels')

# 繪製 SVM 散佈圖 
ax[0][2].scatter(X_iso[:, 0], X_iso[:, 1], c = predicted)
ax[0][2].set_title('SVM_Predicted labels')
ax[1][2].scatter(X_iso[:, 0], X_iso[:, 1], c = y_train)
ax[1][2].set_title('SVM_Actual Labels')

# 顯示圖形
plt.show()
'''
