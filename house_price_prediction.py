# HOUSE PRICE PREDICTION - ML FINAL PROJECT CODE
######################################

# TABLE OF CONTENT
######################################
# 1 - SETTING UP THE ESSENTIALS AND IMPORTING THE DATASET
# 2 - PREVIEW OF THE DATASET
# 3 - DATASET PREPARATION
# 4 - MODELLING
# 5 - MODEL TESTING

######################################
# 1 - SETTING UP THE ESSENTIALS AND IMPORTING THE DATASET
######################################

import pandas as pd
import warnings
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from helpers.data_prep import *
from helpers.eda import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv(r"C:\Users\omerb\Desktop\Python files\House Price Prediction Project\house_price_prediction.csv")
#df = pd.read_csv(r"/Users/evrimkarabati/PycharmProjects/pythonProject/Final_Project/house_price_prediction.csv")

######################################
# 2 - PREVIEW OF THE DATASET
######################################
# Dataset at a glance
df.head()
check_df(df)

######################################
# 2.1 - SNAPSHOT OF THE TARGET VARIABLE
######################################
df["SalePrice"].describe().T
df['SalePrice'].mean()
sns.distplot(df.SalePrice)
plt.show(block=True)

sns.boxplot(df["SalePrice"])
plt.show(block=True)

######################################
# 3 - DATASET PREPARATION
######################################

##################################
# 3.1 - CLASSIFICATION OF THE VARIABLE TYPES(CATEGORICAL AND NUMERICAL)
##################################
# Changing data type of MSSubClass column to string since it is not a numerical value originally, but categorical value.
# We dont want it to be classified as numerical.
df['MSSubClass'] = df['MSSubClass'].astype(str)

def grab_col_names(dataframe, cat_th=3, car_th=10):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, cat_but_car, num_cols, num_but_cat

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

# Observations: 1460
# Variables: 81
# cat_cols: 53
# num_cols: 27
# cat_but_car: 1
# num_but_cat: 11

######################################
# 3.2 - ADJUSTING MISSING VALUE COLUMNS
######################################

# NA values in some columns represent that the observed unit does not have that feature.
# Since some of those columns already have 'No' as a value, we will fill the NA values with string 'NA' to represent that the feature is not available.
no_cols = ["Alley","BsmtQual",'BsmtCond',"BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

for col in no_cols:
    df[col].fillna("NA", inplace=True)


# Transforming ordinal categorical columns to numeric in order to have meaningful rating values
numeric = ["ExterQual","ExterCond","BsmtCond","BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC","BsmtQual","KitchenQual","FireplaceQu", "PoolQC", 'Fence'
"GarageQual","GarageCond", "GarageCars","OverallCond","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr", "KitchenAbvGr","Fireplaces","PoolArea"]

ratings = {"Ex":5, "Gd":4, "TA":3, 'Fa':2, 'Po':1, 'NA':0}
df['ExterQual'] = df['ExterQual'].replace(ratings)
df['ExterCond'] = df['ExterCond'].replace(ratings)
df['BsmtCond'] = df['BsmtCond'].replace(ratings)
df['BsmtQual'] = df['BsmtQual'].replace(ratings)
df['HeatingQC'] = df['HeatingQC'].replace(ratings)
df['GarageQual'] = df['GarageQual'].replace(ratings)
df['GarageCond'] = df['GarageCond'].replace(ratings)
df['KitchenQual'] = df['KitchenQual'].replace(ratings)
df['FireplaceQu'] = df['FireplaceQu'].replace(ratings)
df['PoolQC'] = df['PoolQC'].replace(ratings)

basement_rating= {"Ex":5, "Gd":4, "Av":3, 'Mn':2, 'No':1, 'NA':0}
df['BsmtExposure'] = df['BsmtExposure'].replace(basement_rating)

basement_rating2= {"GLQ":7, "ALQ":6, "Av":5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0}
df['BsmtFinType1'] = df['BsmtFinType1'].replace(basement_rating2)
df['BsmtFinType2'] = df['BsmtFinType2'].replace(basement_rating2)

fence_rating= {"ExPrv":5, 'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1, 'NA':0}
df['Fence'] = df['Fence'].replace(fence_rating)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

# To observe the new classification of numerical and categorical columns:
#print(f"Observations: {df.shape[0]}")
#print(f"Variables: {df.shape[1]}")
#print(f'cat_cols: {len(cat_cols)}')
#print(f'num_cols: {len(num_cols)}')
#print(f'cat_but_car: {len(cat_but_car)}')
#print(f'num_but_cat: {len(num_but_cat)}')

######################################
# 3.3 - CATEGORICAL VARIABLE ANALYSIS
######################################
# cat_summary fonksiyonunda ufak bir değişiklik categorik değişkenlerin sınıfı dagılımı ile sales price içerisindeki yogunlugunu da yandaki grafikte göstermektedi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        fig , axes = plt.subplots(1, 2, figsize=(15, 8))
        sns.countplot(x=dataframe[col_name], data=dataframe,  ax=axes[0])
        sns.histplot(dataframe, x=df["SalePrice"], hue=dataframe[col_name], ax=axes[1])
        fig.suptitle(col_name + " " + "vs SalesPrice", fontsize=16)
        plt.xticks(rotation=90)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

######################################
# 3.4 - NUMERICAL VARIABLE ANALYSIS
######################################
for col in num_cols:
    num_summary(df, col)

######################################
# 3.5 - TARGET VALUE ANALYSIS
######################################
for col in cat_cols:
    target_summary_with_cat(df,"SalePrice",col)

######################################
# 3.6 - OUTLIER ANALYSIS
######################################
# Checking outliers in each numeric column
for col in num_cols:
    if col != "SalePrice":
      print(col, check_outlier(df, col))

# Replacing the outliers
for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)

######################################
# 3.6 - MISSING VALUE ANALYSIS
######################################
# Finding missing values after filling NA values with 'NA' in 3.2.
na_columns = missing_values_table(df, na_name=True)

# NaN values per column.
#             n_miss  ratio
# LotFrontage     259 17.740
# GarageYrBlt      81  5.550
# MasVnrType        8  0.550
# MasVnrArea        8  0.550
# Electrical        1  0.070

# Missing Value Columns VS Target Column :
missing_vs_target(df, "SalePrice", na_columns)

#                      TARGET_MEAN  Count
# LotFrontage_NA_FLAG
# 0                     180770.480   1201
# 1                     181620.073    259
#
#                    TARGET_MEAN  Count
# MasVnrType_NA_FLAG
# 0                    180615.063   1452
# 1                    236484.250      8
#
#                    TARGET_MEAN  Count
# MasVnrArea_NA_FLAG
# 0                    180615.063   1452
# 1                    236484.250      8
#
#                    TARGET_MEAN  Count
# Electrical_NA_FLAG
# 0                    180930.395   1459
# 1                    167500.000      1
#                     TARGET_MEAN  Count
# GarageYrBlt_NA_FLAG
# 0                     185479.511   1379
# 1                     103317.284     81


# Filled NA values of LotFrontage according to MSSubClass group by. As LotFrontage & MSSubClass have high correlation.
df["LotFrontage"].value_counts().sort_values(ascending=False)
df["LotFrontage"].isnull().sum()
df['LotFrontage'] = df.groupby(['MSSubClass'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

# GarageYrBlt is very highly correlated with YearBuilt, dropped GarageYrBlt column to overcome missing values in it.
df.drop('GarageYrBlt',axis=1,inplace=True)

# Dropping NA's for columns with low NA count. Total of 9 rows are dropped out of 1460.
mis_cols = ['Electrical', 'MasVnrArea', 'MasVnrType']
df.dropna(subset=mis_cols, inplace=True)

######################################
# 3.7 - RARE ENCODING
######################################
rare_analyser(df, "SalePrice", cat_cols)

# LotShape : 4
#      COUNT  RATIO  TARGET_MEAN
# IR1    968  0.332   206101.665
# IR2     76  0.026   239833.366
# IR3     16  0.005   216036.500
# Reg   1859  0.637   164754.818

df["LotShape"] = np.where(df.LotShape.isin(["IR1", "IR2", "IR3"]), "IR", df["LotShape"])

# Combining RARE classes together that has less than 0.01 ratio.
df = rare_encoder(df,0.01)

######################################
# 3.8 - FEATURE ENGINEERING
######################################
# Creating new columns to add more dimension to our analysis.
df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                      "BsmtFinType2", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis = 1)

df["TotalGarageQual"] = df[["GarageQual", "GarageCond"]].sum(axis = 1)

df["Overall"] = df[["OverallQual", "OverallCond"]].sum(axis = 1)

df["Exter"] = df[["ExterQual", "ExterCond"]].sum(axis = 1)

df["Qual"] = df[["OverallQual", "ExterQual", "GarageQual", "Fence", "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu"]].sum(axis = 1)

df["Cond"] = df[["OverallCond", "ExterCond", "GarageCond", "BsmtCond", "HeatingQC", "Functional"]].sum(axis = 1)

# Adding area columns that are related.
# Total Floor
df["TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]

# Total Finished Basement Area
df["TotalBsmtFin"] = df.BsmtFinSF1+df.BsmtFinSF2

# Porch Area
df["PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF

# Total House Area
df["TotalHouseArea"] = df.TotalFlrSF + df.TotalBsmtSF

df["TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF

df["TotalFullBath"] = df.BsmtFullBath + df.FullBath
df["TotalHalfBath"] = df.BsmtHalfBath + df.HalfBath
df["TotalBath"] = df["TotalFullBath"] + (df["TotalHalfBath"]*0.5)

# Creating ratio columns.
# Lot Ratio
df["LotRatio"] = df.GrLivArea / df.LotArea

df["RatioArea"] = df.TotalHouseArea / df.LotArea

df["GarageLotRatio"] = df.GarageArea / df.LotArea

# MasVnrArea
df["MasVnrRatio"] = df.MasVnrArea / df.TotalHouseArea

# More meaningful columns are added to enhance our model
# Dif Area
df["DifArea"] = (df.LotArea - df["1stFlrSF"] - df.GarageArea - df.PorchArea - df.WoodDeckSF)

# LowQualFinSF
df["LowQualFinSFRatio"] = df.LowQualFinSF / df.TotalHouseArea

# OverallGrade
df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]

# Overall kitchen score
df["KitchenScore"] = df["KitchenAbvGr"] * df["KitchenQual"]

# Overall fireplace score
df["FireplaceScore"] = df["Fireplaces"] * df["FireplaceQu"]

df["Restoration"] = df.YearRemodAdd - df.YearBuilt
df["HouseAge"] = df.YrSold - df.YearBuilt
df["RestorationAge"] = df.YrSold - df.YearRemodAdd

# Has Columns(whether the house has the feature)
df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Dropping some columns that have little or no contribution to our model
drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature"]
df.drop(drop_list, axis=1, inplace=True)

######################################
# 3.9 - LABEL ENCODING - ONE-HOT ENCODING
######################################
# Redefining numerical and categorical columns after adding new columns
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)
cat_cols= cat_cols + cat_but_car
num_cols = [col for col in num_cols if col not in ['Id', 'SalePrice']]

#Observations: 1451
#Variables: 103
#cat_cols: 25
#num_cols: 74
#cat_but_car: 4
#num_but_cat: 6

# Label Encoding
binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

binary_cols

for col in binary_cols:
    label_encoder(df, col)

# One-Hot Encoding
df = one_hot_encoder(df, cat_cols, drop_first=True)

######################################
# 3.10 - STANDARD SCALING
######################################
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

######################################
# 4 - MODELLING
######################################

######################################
# 4.1 - DIVIDING THE DATA SET(TRAIN - TEST)
######################################

y = df["SalePrice"]
X = df.drop(["Id", "SalePrice"], axis=1)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
y_train = np.ravel(y_train)

######################################
# 4.2 - INITIAL MODEL ALTERNATIVES
######################################

# Creating the list of the ML models
models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          #("CatBoost", CatBoostRegressor(verbose=False))]

# Finding RMSE values for each model
for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


df['SalePrice'].mean()
df['SalePrice'].std()


######################################################
# 4.3 - AUTOMATED HYPERPARAMETER OPTIMIZATION
######################################################

# Setting parameters for each model
cart_params = {'max_depth': range(5, 15),
               "min_samples_split": range(15, 30)}

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [5, 15, 20],
             "n_estimators": [100, 200, 500]}

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 200, 300],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.1, 0.01, 0.001],
                   "n_estimators": [300, 500, 1000],
                   "colsample_bytree": [0.5, 0.7, 1]}

gbm_params = {"learning_rate": [0.1, 0.01, 0.001],
              "max_depth": [3, 8, 12],
              "n_estimators": [300, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

regressors = [("CART", DecisionTreeRegressor(), cart_params),
              ("RF", RandomForestRegressor(), rf_params),
              ("GBM", GradientBoostingRegressor(), gbm_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ]

best_models = {}


# Defining feature importance function
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title(model)
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

# Running hyperparameter optimization
for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE:(Before with train) {round(rmse, 4)} ({name}) ")

    # Building the model with best parameters.
    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X_train, y_train)
    final_model = regressor.set_params(**gs_best.best_params_).fit(X_train, y_train)

    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE (After with train): {round(rmse, 4)} ({name})")
    print(f"{name} best params: {gs_best.best_params_}")

    # Predicting y values based on X_test
    y_pred = final_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE (After with test): {round(rmse, 4)} ({name}) ", end="\n\n")

    best_models[name] = final_model
    #plot_importance(final_model, X, 20)

#We pick XGBoost based on the RMSE values
print(final_model)
print(best_models)

########## XGBoost ##########
#RMSE:(Before with train) 32738.3549 (XGBoost)
#RMSE (After): 30057.2409 (XGBoost)
#XGBoost best params: {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
#RMSE (After with test): 23507.7456 (XGBoost)

########## GBM ##########
#RMSE:(Before with train) 30352.6605 (GBM)
#RMSE (After with train): 29637.3102 (GBM)
#GBM best params: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 0.5}
#RMSE (After with test): 21594.6218 (GBM)

########## LightGBM ##########
#RMSE:(Before with train) 30487.41 (LightGBM)
#RMSE (After with train): 29340.1343 (LightGBM)
#LightGBM best params: {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 500}
#RMSE (After with test): 23872.278 (LightGBM)

######################################################
# 4.4 - MODEL TUNING AND FINALIZING THE MODEL
######################################################
# RMSE for train data with base model.
rmse = np.mean(np.sqrt(-cross_val_score(XGBRegressor(), X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
print(f"RMSE (Before with train): {round(rmse, 4)} ")

# Model Tuning
xgb_params = {"learning_rate": [0.1, 0.05, 0.01, 0.001],
             "max_depth": [1, 5, 8, 15],
             "n_estimators": [100, 200, 300, 500],
             "colsample_bytree": [0.1, 0.3, 0.5, 0.9]}
# The RMSE results of the tuned XGB with the best parameters.
#RMSE (After with train): 27157.3236
#RMSE (After with test): 20268.6408
#XGBoost best params: {'colsample_bytree': 0.1, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 500}

gbm_params = {"learning_rate": [0.1, 0.05, 0.01, 0.001],
              "max_depth": [3, 5, 8, 12],
              "n_estimators": [300, 400, 500, 700],
              "subsample": [0.2, 0.5, 0.7, 0.9]}

# The RMSE results of the tuned GBM with the best parameters.
#RMSE (After with train): 29744.088
#RMSE (After with test): 21461.4616
#GBM best params: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.5}

lightgbm_params = {"learning_rate": [0.1, 0.05, 0.01, 0.001],
                   "n_estimators": [200, 400, 500, 700],
                   "colsample_bytree": [0.1, 0.3, 0.5, 0.9]}

# The RMSE results of the tuned LightGBM with the best parameters.
#RMSE (After with train): 28995.0637
#RMSE (After with test): 22328.361
#Lightgbm best params: {'colsample_bytree': 0.1, 'learning_rate': 0.01, 'n_estimators': 700}

for params in xgb_params:
    # Finding best params
    xgb_cv_model = GridSearchCV(XGBRegressor(), xgb_params, cv=3, n_jobs=-1, verbose=False).fit(X_train, y_train)
    xgb_cv_model.best_params_

    # Final Model
    xgb_tuned = XGBRegressor(**xgb_cv_model.best_params_).fit(X_train, y_train)

    rmse = np.mean(np.sqrt(-cross_val_score(xgb_tuned, X_train, y_train, cv=3, scoring="neg_mean_squared_error")))
    print(f"RMSE (After with train): {round(rmse, 4)}")

    # Final Model Verification with Test Dataset
    y_pred = xgb_tuned.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE (After with test): {round(rmse, 4)}")
    print(f"XGBoost best params: {xgb_cv_model.best_params_}", end="\n\n")

######################################################
# 4.5 - STACKING & ENSEMBLE LEARNING
######################################################
# Combining models to improve the RMSE score. In this case the best performers are GBM, LightGBM and XGBoost.
def voting_regressor(best_models, X_train, y_train, X_test, y_test):
    voting_rg = VotingRegressor(estimators=[('XGBoost', best_models["XGBoost"]),
                                              ('LightGBM', best_models["LightGBM"]), ('GBM', best_models["GBM"])]).fit(X_train, y_train)

    rmse = np.mean(np.sqrt(-cross_val_score(voting_rg, X_train, y_train, cv=3, scoring="neg_mean_squared_error")))
    print(f"RMSE (After with train):{round(rmse, 4)}")

    y_pred = voting_rg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE (After with test): {round(rmse, 4)}")

    return voting_rg

voting_reg = voting_regressor(best_models, X_train, y_train, X_test, y_test)

#RMSE (After with train): 28576.3205
#RMSE (After with test): 21351.9896

# Voting_reg with best params manually entered
new_best_models={'XGBoost': XGBRegressor(colsample_bytree=0.1, learning_rate=0.05, max_depth=5, n_estimators=500), 'GBM': GradientBoostingRegressor(n_estimators=500, subsample=0.5, max_depth=3, learning_rate=0.1), 'LightGBM': LGBMRegressor(colsample_bytree=0.1, learning_rate=0.01, n_estimators=700)}
voting_reg = voting_regressor(new_best_models, X_train, y_train, X_test, y_test)
#RMSE (After with train):27232.5237
#RMSE (After with test): 20534.6605

######################################################
# 5 - MODEL TESTING
######################################################
# Prediction of a new observation
X.columns
random_user = X.sample(1, random_state=45)
xgb_tuned.predict(random_user)

#RMSE
y_test_pred = xgb_tuned.predict(X_test)
mse = mean_squared_error(y_test_pred, y_test)
rmse = np.sqrt(mse)

print("RMSE : % f" %(rmse))