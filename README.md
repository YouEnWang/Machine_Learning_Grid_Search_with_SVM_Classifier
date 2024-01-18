# 目標
利用鳶尾花資料(Iris data set)來訓練Support Vector Machine (SVM)，採用one-against one strategy來處理三類別分類問題，並以grid search來最佳化SVM之參數C與sigma，其過程再以two-fold cross validation使所得到的最佳參數組具有較佳的泛化性。

# 資料描述
1. 安德森鳶尾花卉數據集(Anderson's Iris data set)為機器學習領域中，常被用來驗證演算法效能的資料庫。數據庫中包含三種不同鳶尾花標籤(Label)：山鳶尾(Setosa)、變色鳶尾(Versicolor)以及維吉尼亞鳶尾(Virginica)，且每種鳶尾花皆有50筆樣本。而每筆樣本以四種屬性作為特徵(單位：cm)：花萼長度(Sepal length)、花萼寬度(Sepal width)、花瓣長度(Petal length)以及花瓣寬度(Petal width)等四種屬性作為定量分析之數據。
2. 讀取鳶尾花資料後會產生150×5的陣列，其中第5行為資料的類別標籤。

# 作業內容
1. 將Iris data set的山鳶尾(Setosa)、變色鳶尾(Versicolor)以及維吉尼亞鳶尾(Virginica)各別取前25筆data設為training data，剩餘的75筆設為test data，所有資料皆採用全部4個特徵，並採用同一組SVM參數(C與sigma)建立三類別分類器。
2. 使用2-fold CV進行交叉驗證。
3. 改用下一組SVM參數(C與sigma)重複上述步驟，直到所有參數組合(7×41組)皆完成測試。

## 備註
其中，此作業皆採用RBF-kernel之SVM，且grid search範圍如下：
C: 1, 5, 10, 50, 100, 500, 1000
sigma: 1.05E-100, 1.05E-95, …, 1.05E-10, 1.05E-05, 1.050, 1.05E+05, 1.05E+10, …, 1.05E+95, 1.05E+100

# 程式執行方式
- 直接執行程式即可產生結果
- 根據penalty weight的不同，會產生6個不同的csv檔，其中紀錄了當前的C下，不同的sigma所產生的不同分類率

# 討論
1. 請問在grid search的結果中，C的大小與分類率高低有何關係?
	- 從結果來說，當C愈大時分類率愈高。因為當C愈大時，模型會盡力擬合每一個訓練樣本點。將C增大會使模型更嚴格的去嘗試擬合每個訓練資料，雖然可能會因此讓模型更容易受 noise 影響而發生過擬合，但由於 Iris data 的 feature 不多且數據的複雜度不高，所以預測結果不太會受 noise 影響。

2. sigma 的大小的改變與分類率是否有關係? 若有，請探討 sigma 的差異與特徵的數值有什麼關聯性?
	- 當 sigma 值太小時，高斯分布的曲線會長的又高又瘦，使得模型較為關注 support vector 附近局部的樣本點，造成模型對於未知樣本點的分類效果會很差。當 sigma 值愈大，模型愈能考慮廣泛的全域樣本點，模型會對未知樣本點有更好的分類結果，泛化性能較高。以 Iris data 的4 個特徵皆取的狀況下，特徵點之間的分布範圍較大，適合用較大的sigma 值來做 SVM 分類，grid search 的結果也能映證此事。

3. 若分析過程不採用 two-fold cross validation，則分類率是否會更高? 請探討之。
   - 若不採用 two-fold cross validation，雖然有些參數組合會產生較高的分類率，但並非整體的分類率都有提高。如果沒有進行交叉驗證，進行 grid search 時可能會受數據分割的位置影響。對於此作業之 Iris data而言，各個 class 的樣本點有進行打散，所以 feature 較沒有遞增或遞減的趨勢，因此不進行交叉驗證的話並不會對分類率有太大的影響。

4. 結果分析
   - 對於 RBF kernel 而言，隨著 sigma 增加，RBF kernel 的超平面會更貼近 support vector，可能導致過擬合。這解釋了 grid search 的結果中，sigma 增加到某一數值時，分類率反而降低的現象。這次作業的 k-fold cross validation 和 grid search，都是機器學習中常用來避免超平面發生過擬合的方法，而分類率結果顯示，兩種方法結合可以實際的避免overfitting，找出最佳的參數組合。

# 心得
這次的作業難度較不高，但我出乎意料的花了很久的時間。起初嘗試很多次的分類率結果都低於 80%，後來仔細的去比對數據以及第一次作業 KNN 的結果，才發現原來是計算 RBF kernel 的 function 中，我沒有依照 feature 的數量去更改norm 的計算式。此外，由於我在設計 HW3 的 SVM 模型時，是針對 HW3 的作業內容去設計，並沒有設計出較為通用的 SVM 模型，所以在設計 HW4 的多類別分類時，很多 function 的 parameters 跟 return value 都要重新思考。雖然花了很多時間在修改單類別 SVM 分類模型，但也讓我更了解 SVM 的演算法設計。
