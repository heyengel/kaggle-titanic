import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

#define crossvalidation function
def crossValidate(features, target, classifier, k_fold, r_state=None) :
    # derive a set of (random) training and testing indices
    k_fold_indices = KFold(len(features), n_folds=k_fold,
                           shuffle=True, random_state=r_state)
    
    # for each set of training and testing indices 
    # train the classifier, and score the results
    k_score_total = 0
    for train_indices, test_indices in k_fold_indices :

        model = classifier.fit(features[train_indices],
                           target[train_indices])

        k_score = model.score(features[test_indices],
                              target[test_indices])

        k_score_total = k_score_total + k_score

    # return the average accuracy
    return k_score_total/k_fold

# import data
df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# clean the data
median_age = df['Age'].median()
avg_fare = df['Fare'].mean()

df['Age'].fillna(median_age, inplace = True)
df['Embarked'].fillna('S', inplace = True)
df['Fare'].fillna(avg_fare, inplace = True)
test['Age'].fillna(median_age, inplace = True)
test['Fare'].fillna(avg_fare, inplace = True)

# map features
df['Gender'] = df['Sex'].map({'male': 1, 'female': 0})
df['FamilySize'] = df['Parch'] + df['SibSp']
df['EmbarkPort'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'nan': 0})

test['Gender'] = test['Sex'].map({'male': 1, 'female': 0})
test['FamilySize'] = df['Parch'] + df['SibSp']
test['EmbarkPort'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'nan': 0})


# define features (X) and target (y) variables
#X = df[['Pclass', 'Age', 'Gender']]
X = df[['Pclass', 'Age','Fare', 'Gender', 'EmbarkPort', 'FamilySize']]
y = df['Survived']

X2 = test[['Pclass', 'Age','Fare', 'Gender', 'EmbarkPort', 'FamilySize']]

features = X.values
target = y.values

# Initialize the model
model = RandomForestClassifier(25)


#  Split the features and the target into a Train and a Test subsets.  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size = 0.2, random_state=0)

# Train the model
model.fit(X_train, y_train)


# Calculate the model score
my_score = model.score(X_test, y_test)

print "\n"
print "Using model: %s" % model
print "Classification Score: %0.2f" % my_score

# Print the confusion matrix for the decision tree model
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
print "\n=======confusion matrix=========="
print confusion_matrix(y_test, y_pred)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": model.predict(X2)
    })

submission.to_csv('submission.csv', index=False)
