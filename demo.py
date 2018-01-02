from sklearn import tree

clf = tree.DecisionTreeClassifier()

# [tricks learnt, weight in lbs, number of friends]
X = [[5, 70, 10], [8, 90, 30],
     [1, 30, 1], [2, 44, 2],
     [5, 100, 36], [7, 90, 42],
     [1, 34, 0], [4, 36, 2], [2, 25, 1],
     [8, 55, 66], [6, 68, 22]]

Y = ['dog', 'dog', 
    'cat', 'cat', 
    'dog', 'dog', 
    'cat', 'cat','cat', 
    'dog', 'dog']

clf = clf.fit(X, Y)

prediction = clf.predict([[6, 35, 4]])

print(prediction)