from sklearn.linear_model import LinearRegression

def lr_predict(x_train, y_train, x_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    return lr.predict(x_test)
