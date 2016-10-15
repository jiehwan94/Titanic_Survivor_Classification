---
title: "Titanic_Survivor_Classification"
author: "Jiehwan Yang (����ȯ)"
date: "2016�� 10�� 15��"
output: html_document
---





# Ÿ��Ÿ�� ������ ������

## Loading data...

�� ������ �ǹ��ϴ� ���� īī���忡 �ø� �� �����غ�����~

```
##   PassengerId Survived Pclass                                                Name    Sex Age SibSp Parch
## 1           1        0      3                             Braund, Mr. Owen Harris   male  22     1     0
## 2           2        1      1 Cumings, Mrs. John Bradley (Florence Briggs Thayer) female  38     1     0
## 3           3        1      3                              Heikkinen, Miss. Laina female  26     0     0
## 4           4        1      1        Futrelle, Mrs. Jacques Heath (Lily May Peel) female  35     1     0
## 5           5        0      3                            Allen, Mr. William Henry   male  35     0     0
## 6           6        0      3                                    Moran, Mr. James   male  NA     0     0
##             Ticket    Fare Cabin Embarked
## 1        A/5 21171  7.2500              S
## 2         PC 17599 71.2833   C85        C
## 3 STON/O2. 3101282  7.9250              S
## 4           113803 53.1000  C123        S
## 5           373450  8.0500              S
## 6           330877  8.4583              Q
```

```
## [1] 891  12
```

## Data pre-processing...

Categorical�� �� �������� ���� ���߿� decision tree algorithm�� parameter�� ���� �� ������ �߻��ϹǷ� numerical value�� �ٲ���� �մϴ�.

```r
# Data pre-processing...
# Currently, the data contain categorical values.
# Since the decision tree library we will use later does not 
#like categorical values, we need to convert the categorical values 
#to numerical values as follows...
data$Sex_Code[data$Sex=="male"]<- "1"
data$Sex_Code[data$Sex=="female"]<- "2"

data$Embarked_Code[data$Embarked=="S"]<- "1"
data$Embarked_Code[data$Embarked=="c"]<- "2"
data$Embarked_Code[data$Embarked=="Q"]<- "3"

head(data)
```

```
##   PassengerId Survived Pclass                                                Name    Sex Age SibSp Parch
## 1           1        0      3                             Braund, Mr. Owen Harris   male  22     1     0
## 2           2        1      1 Cumings, Mrs. John Bradley (Florence Briggs Thayer) female  38     1     0
## 3           3        1      3                              Heikkinen, Miss. Laina female  26     0     0
## 4           4        1      1        Futrelle, Mrs. Jacques Heath (Lily May Peel) female  35     1     0
## 5           5        0      3                            Allen, Mr. William Henry   male  35     0     0
## 6           6        0      3                                    Moran, Mr. James   male  NA     0     0
##             Ticket    Fare Cabin Embarked Sex_Code Embarked_Code
## 1        A/5 21171  7.2500              S        1             1
## 2         PC 17599 71.2833   C85        C        2          <NA>
## 3 STON/O2. 3101282  7.9250              S        2             1
## 4           113803 53.1000  C123        S        2             1
## 5           373450  8.0500              S        1             1
## 6           330877  8.4583              Q        1             3
```

## Partitioning data...

�����͸� 50�� 50���� Ʈ���̴� �°� �׽�Ʈ ������ �����ݴϴ�.

```r
# Partitioning data into training and test dataset (50%:50%)
ind = sample(1:nrow(data),floor(nrow(data)*0.50))
data_train = data[ind,]
data_test = data[-ind,]

print (c(length(data$PassengerId),length(data_train$PassengerId),length(data_train$PassengerId)))
```

```
## [1] 891 445 445
```



```r
# Load rpart and rpar.plot library iun order to perform prediction analysis by using decision tree
library(rpart)
library(rpart.plot)
```

�켱 minsplit�� 10�� �����ְ� decision tree model �� ����� �ݴϴ�.
���⼭ minsplit=10�� ������ ���� 10�� �̸��� ��, �� node���� split���� �ʵ��� ���ݴϴ�.

```r
# I will split the data at min split value of 10
# Minsplit says don't split if you have fewer than 10 observations left in the branch from the training data
tree_1<- rpart(Survived ~ Pclass+ Age+ SibSp+ Parch+ Fare+ Sex_Code+ Embarked_Code,
               data=data_train, method="class", control=rpart.control(minsplit=10))
# Draw decision tree using rpart.plot
rpart.plot(tree_1, type=1, extra=2, under=TRUE)
title(sub="Decision Tree with minsplit= 10")
```

![plot of chunk unnamed-chunk-4](Figs/unnamed-chunk-4-1.png)

�̹��� minsplit�� 20���� �����س��� ���� �����մϴ�.

```r
# Let's try it with minsplit 20 this time.
tree_2<- rpart(Survived ~ Pclass+ Age+ SibSp+ Parch+ Fare+ Sex_Code+ Embarked_Code,
               data=data_train, method="class", control=rpart.control(minsplit=20))

rpart.plot(tree_2, type=1, extra=2, under=TRUE)
title(sub="Decision Tree with minsplit= 20")
```

![plot of chunk unnamed-chunk-5](Figs/unnamed-chunk-5-1.png)

������ �� model�� ������� ���غ��ô�.

```r
data_test1<- data_test
data_test2<- data_test

data_test1$predicted<- predict(tree_1,data_test,type="class")
data_test2$predicted<- predict(tree_2,data_test,type="class")

confusionmatrix1<- table(data_test1$Survived,data_test1$predicted,dnn=c("Actual","Predicted"))
confusionmatrix2<- table(data_test2$Survived,data_test2$predicted,dnn=c("Actual","Predicted"))

confusionmatrix1
```

```
##       Predicted
## Actual   0   1
##      0 239  19
##      1  84 104
```

```r
confusionmatrix2
```

```
##       Predicted
## Actual   0   1
##      0 239  19
##      1  84 104
```



```r
Accuracy1<-(confusionmatrix1[1,1]+confusionmatrix1[2,2])/sum(confusionmatrix1)
Precision_N1<-confusionmatrix1[1,1]/(confusionmatrix1[1,1]+confusionmatrix1[2,1])
Precision_P1<-confusionmatrix1[2,2]/(confusionmatrix1[2,2]+confusionmatrix1[1,2])
Recall_N1<-confusionmatrix1[1,1]/(confusionmatrix1[1,1]+confusionmatrix1[1,2])
Recall_P1<-confusionmatrix1[2,2]/(confusionmatrix1[2,2]+confusionmatrix1[2,1])

Accuracy2<-(confusionmatrix2[1,1]+confusionmatrix2[2,2])/sum(confusionmatrix2)
Precision_N2<-confusionmatrix2[1,1]/(confusionmatrix2[1,1]+confusionmatrix2[2,1])
Precision_P2<-confusionmatrix2[2,2]/(confusionmatrix2[2,2]+confusionmatrix2[1,2])
Recall_N2<-confusionmatrix2[1,1]/(confusionmatrix2[1,1]+confusionmatrix2[1,2])
Recall_P2<-confusionmatrix2[2,2]/(confusionmatrix2[2,2]+confusionmatrix2[2,1])

print(c(Accuracy1,Accuracy2))
```

```
## [1] 0.7690583 0.7690583
```

```r
print(c(Precision_N1, Precision_N2))
```

```
## [1] 0.7399381 0.7399381
```

```r
print(c(Precision_P1, Precision_P2))
```

```
## [1] 0.8455285 0.8455285
```

```r
print(c(Recall_N1, Recall_N2))
```

```
## [1] 0.9263566 0.9263566
```

```r
print(c(Recall_P1, Recall_P2))
```

```
## [1] 0.5531915 0.5531915
```

minsplit �� ���� ���� ��������� ���� �ٸ� ���� Ȯ���� �� �ֽ��ϴ�. ������ ���ߵ���, ���̳� ������ �����ϴ� ���, FP (������ �Ͼ�µ� �� �Ͼ�ٰ� ������ ���)�� ���� ���� �����ϴ°� �߿��ϴٰ� �� �� �ְ���.

# �ٵ� ���� ȭ���� �ϼ���~ 
