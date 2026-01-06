from sklearn.linear_model import Lasso,Ridge,LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def evaluate(true,predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true,predicted)
    rmse=np.sqrt(mean_squared_error(true,predicted))
    r2=r2_score(true,predicted)
    return mae,mse,rmse,r2


models={
    "LR":LinearRegression(),
    "Lasso":Lasso(),
    "Ridge":Ridge(),
    "KNN":KNeighborsRegressor(),
    "GB":GradientBoostingRegressor(),
    "DT":DecisionTreeRegressor(),
    "RF":RandomForestRegressor(),
    "XGB":XGBRegressor(),
    "ADB":AdaBoostRegressor()
}


for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train,y_train)

    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)
    ##y_pred = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1))
    model_train_mae,model_train_mse,model_train_rmse,model_train_r2=evaluate(y_train,y_train_pred)
    model_test_mae,model_test_mse,model_test_rmse,model_test_r2=evaluate(y_test,y_test_pred)


    print(model)
    print("Model Performance for training set")
    print('-mae:{:.4f}'.format(model_train_mae))
    print('-mse:{:4f}'.format(model_train_mse))
    print('-rmse:{:4f}'.format(model_train_rmse))
    print('-r2:{:4f}'.format(model_train_r2))
    print('-'*35)
    print("Model Performance for testing set")
    print('-mae:{:.4f}'.format(model_test_mae))
    print('-mse:{:4f}'.format(model_test_mse))
    print('-rmse:{:4f}'.format(model_test_rmse))
    print('-r2:{:4f}'.format(model_test_r2))
    print('='*35)
    print('\n')



