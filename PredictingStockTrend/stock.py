import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import seaborn as sns
import sklearn.model_selection
import matplotlib.pyplot as plt

#Encoding are in general UTF-8(default setting), Latin-1 (also known as ISO-8859-1) or Windows-1251
df=pd.read_csv('Combined_News_DJIA.csv', encoding = "ISO-8859-1")

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']
print("Train: ",len(train))
print("Test: ",len(test))


# Removing punctuations
data=train.iloc[:,2:27]
#print(data)
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
#data.head(5)

# Convertng headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()
#data.head(1)

headlines = []
#print(data.index)
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

#print(headlines[0])

## implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)
#print(traindataset[0])

# implement RandomForest Classifier
print(train['Label'])
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])

## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
print(test_transform)
print(len(test_transform))

test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)
print(predictions)


matrix=confusion_matrix(test['Label'],predictions)
#print(matrix)
score=accuracy_score(test['Label'],predictions)
print("Accuracy: ",score*100,"%")
report=classification_report(test['Label'],predictions)
print(report)



con=sklearn.metrics.confusion_matrix(test['Label'],predictions)
ax=plt.subplot()
sns.heatmap(con,annot=True)
plt.title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Down","Up"])
ax.yaxis.set_ticklabels(["Down","Up"])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()
