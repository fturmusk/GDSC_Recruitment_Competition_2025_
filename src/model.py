import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor


def best_model_(train_final,y_train):
    x_train, x_test, Y_train, y_test = train_test_split(train_final,y_train, train_size=0.75, random_state=42)

    linear_model = LinearRegression()

    random_model = RandomForestRegressor(
        n_estimators=200,
        criterion='squared_error',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    )

    gauss_model = GaussianProcessRegressor()

    xgb_model = XGBRegressor()

    lgb_model = LGBMRegressor(
            random_state=42,
            verbosity=-1,
            n_estimators=40000,
            learning_rate=0.0358306214515723,
            min_child_samples=83,
            subsample=0.8700304020753131,
            colsample_bytree=0.6169349166144594,
            num_leaves=228,
            max_depth=6,
            max_bin=3600,
            reg_alpha=3.700714656885025,
            reg_lambda=4.709578317972932, 
        )

    models = [linear_model,random_model,gauss_model,xgb_model,lgb_model]

    model_saved = []
    for model in models:
        model.fit(x_train, Y_train)

        #train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        model_saved.append(mean_absolute_error(y_test,test_pred))
    
    best_model = model_saved.index(min(model_saved))
    return best_model

def test_predict_and_submit(best_model,train_final,y_train, test_final,df__test):
    x_train, x_test, Y_train, y_test = train_test_split(train_final,y_train, train_size=0.75, random_state=42)

    linear_model = LinearRegression()

    random_model = RandomForestRegressor(
        n_estimators=200,
        criterion='squared_error',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    )

    gauss_model = GaussianProcessRegressor()

    xgb_model = XGBRegressor()

    lgb_model = LGBMRegressor(
            random_state=42,
            verbosity=-1,
            n_estimators=40000,
            learning_rate=0.0358306214515723,
            min_child_samples=83,
            subsample=0.8700304020753131,
            colsample_bytree=0.6169349166144594,
            num_leaves=228,
            max_depth=6,
            max_bin=3600,
            reg_alpha=3.700714656885025,
            reg_lambda=4.709578317972932, 
        )

    model = [linear_model,random_model,gauss_model,xgb_model,lgb_model]

    if best_model == 0:
        model[0].fit(x_train, Y_train)
        test_pred = model[0].predict(test_final)
    
    if best_model == 1:
        model[1].fit(x_train, Y_train)
        test_pred = model[1].predict(test_final)
    if best_model == 2:
        model[2].fit(x_train, Y_train)
        test_pred = model[2].predict(test_final)
    if best_model == 3:
        model[3].fit(x_train, Y_train)
        test_pred = model[3].predict(test_final)
    if best_model == 4:
        model[4].fit(x_train, Y_train)
        test_pred = model[4].predict(test_final)
    submission = pd.DataFrame({
        'LOCAL_IDENTIFIER': df__test['LOCAL_IDENTIFIER'],
        'CORRUCYSTIC_DENSITY': test_pred
    })

    submission.to_csv('data/submission.csv', index=False)
    print("Submission saved!")

