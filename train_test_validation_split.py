from sklearn.model_selection import train_test_split

'''produces a 60%, 20%, 20% split for training, validation and test sets. '''
def train_test_validation_split(X, y, train_size=0.6 ,test_size=0.2, val_size=0.2, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size+val_size, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(train_size+val_size), random_state=random_state)
    return X_train, X_test, X_val, y_train, y_test, y_val

X = [i  for i in range(500)]
y= [i**2 for i in range(500)]
X_train, X_test, X_val, y_train, y_test, y_val = train_test_validation_split(X,y, random_state=42)
print(X_train, X_test, X_val)