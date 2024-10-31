from sklearn.model_selection import train_test_split

def shapes(df1, df2):
    print('Shape df1 ' ,df1.shape)
    print('Shape df2 ' ,df2.shape)

def data_split(data):
    X = data.drop('Class', axis=1).copy()
    y = data['Class'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


    print('Training', '-'*20)
    shapes(X_train, y_train)

    print('Validation', '-'*20)
    shapes(X_val, y_val)

    print('Test ', '-'*20)
    shapes(X_test, y_test)
    return X_train, X_test, y_train, y_test, X_val, y_val