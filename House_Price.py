import numpy as np  # linear algebra
import pandas as pd  #
from datetime import datetime

from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score
import xgboost as xg

from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import os
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Deleting outliers
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)

# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
# Salse price is skewed so need log1p

train["SalePrice"] = np.log1p(train["SalePrice"])

y = train.SalePrice.reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test

features = pd.concat([train_features, test_features]).reset_index(drop=True)

# Fill in missing data of the essential features
features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
features['Functional'] = features['Functional'].fillna('Typ')
features['KitchenQual'] = features['KitchenQual'].fillna("TA")
features['Electrical'] = features['Electrical'].fillna("SBrkr")
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

# Filling in the rest of the NA's
numerics2 = []
for i in features.columns:
    if features[i].dtype == object:
        features[i] = features[i].fillna('NA')
    elif (features[i].dtype == 'int64') | (features[i].dtype == 'float64'):
        features[i] = features[i].fillna(0)
        numerics2.append(i)

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------

skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[skew_features > 0.5]

for i in high_skew.index:
     features[i] = np.log1p(features[i])
     
features['HavePool'] = 0
features.loc[ features.PoolArea > 0,'HavePool' ] = 1
features['Have2ndfloor'] = 0
features.loc[ features['2ndFlrSF'] > 0,'Have2ndfloor' ] = 1
features['HaveGarage'] = 0
features.loc[ features.GarageArea > 0,'HaveGarage' ] = 1
features['HaveBSMT'] = 0
features.loc[ features.TotalBsmtSF > 0,'HaveBSMT' ] = 1
features['HaveFireP'] = 0
features.loc[ features.Fireplaces > 0,'HaveFireP' ] = 1
features['HouseYear'] = features['YrSold'].astype(int) - features['YearRemodAdd']
features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']
features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
features['AllSF'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] + features['1stFlrSF'] + features['2ndFlrSF'])
features['AllBathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) + features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))
features['PorchSF'] = (features['OpenPorchSF'] + features['3SsnPorch'] + features['EnclosedPorch'] + features['ScreenPorch'] + features['WoodDeckSF'])

'''
Mapping list 
'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0,
'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,
'Av':4,'Mn':3,'No':0
'''

features['Overallscore']  = features['OverallQual'] + features['OverallCond']
features['ExtrenalScore'] = features['ExterCond'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}) + features['ExterQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
features['BSMTScore'] = features['BsmtCond'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}) + features['BsmtQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}) + features['BsmtExposure'].map({'Gd':5,'Av':4,'Mn':3,'No':0,'NA':0}) + features['BsmtFinType1'].map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0}) +features['BsmtFinType2'].map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0})
features['GarageScore'] = features['GarageCond'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}) + features['GarageQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0})  
features['AllKitchenScore'] = features['KitchenAbvGr'] * features['KitchenQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
features['FP_heat_FenceScore'] = features['Fence'].map({'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'NA':0}) + features['FireplaceQu'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}) + features['HeatingQC'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
features['AllScore'] = features['FP_heat_FenceScore'] + features['AllKitchenScore'] + features['GarageScore'] + features['BSMTScore'] + features['ExtrenalScore'] + features['Overallscore']

# Delete the featutres
#features = features.drop(['OverallQual', 'OverallCond', 'ExterCond','ExterQual', 'BsmtCond', 'BsmtQual','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','GarageCond','GarageQual','KitchenAbvGr','KitchenQual','Fence','FireplaceQu','HeatingQC'], axis=1)
# One hot encoding
final_features = pd.get_dummies(features).reset_index(drop=True)

X = final_features.iloc[:len(y), :]
X_test = final_features.iloc[len(X):, :]


outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])
overfit = ['MSZoning_C (all)'] 
for i in X.columns:
    counts = X[i].value_counts()
    for j in counts:
        if j / len(X) > 0.999 :
            overfit.append(i)

overfit = list(overfit)

X = X.drop(overfit, axis=1).copy()
X_test = X_test.drop(overfit, axis=1).copy()

#lightgbm cv = 10
#X = X[['Id', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'HouseYear', 'YrBltAndRemod', 'TotalSF', 'AllSF', 'AllBathrooms', 'PorchSF', 'Overallscore', 'ExtrenalScore', 'BSMTScore', 'AllKitchenScore', 'FP_heat_FenceScore', 'AllScore', 'MSSubClass_30', 'MSSubClass_90', 'MSZoning_RL', 'MSZoning_RM', 'LotShape_IR1', 'LotShape_Reg', 'LandContour_Bnk', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_IDOTRR', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_OldTown', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Condition1_Artery', 'Condition1_Norm', 'BldgType_1Fam', 'HouseStyle_1.5Fin', 'HouseStyle_SLvl', 'RoofStyle_Gable', 'Exterior1st_BrkFace', 'Exterior1st_HdBoard', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior2nd_VinylSd', 'MasVnrType_BrkFace', 'ExterQual_Gd', 'ExterQual_TA', 'ExterCond_Gd', 'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'BsmtQual_Ex', 'BsmtQual_Fa', 'BsmtCond_Fa', 'BsmtExposure_Gd', 'BsmtFinType1_ALQ', 'HeatingQC_Ex', 'HeatingQC_TA', 'CentralAir_N', 'Electrical_FuseA', 'KitchenQual_Ex', 'KitchenQual_TA', 'Functional_Typ', 'FireplaceQu_Gd', 'FireplaceQu_TA', 'GarageType_Attchd', 'GarageType_Detchd', 'GarageFinish_RFn', 'GarageFinish_Unf', 'PavedDrive_Y', 'Fence_GdWo', 'Fence_NA', 'MoSold_1', 'MoSold_10', 'MoSold_3', 'MoSold_5', 'MoSold_7', 'MoSold_9', 'YrSold_2006', 'YrSold_2007', 'YrSold_2009', 'YrSold_2010', 'SaleType_New', 'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_Normal']]
#X_test = X_test[['Id', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'HouseYear', 'YrBltAndRemod', 'TotalSF', 'AllSF', 'AllBathrooms', 'PorchSF', 'Overallscore', 'ExtrenalScore', 'BSMTScore', 'AllKitchenScore', 'FP_heat_FenceScore', 'AllScore', 'MSSubClass_30', 'MSSubClass_90', 'MSZoning_RL', 'MSZoning_RM', 'LotShape_IR1', 'LotShape_Reg', 'LandContour_Bnk', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_IDOTRR', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_OldTown', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Condition1_Artery', 'Condition1_Norm', 'BldgType_1Fam', 'HouseStyle_1.5Fin', 'HouseStyle_SLvl', 'RoofStyle_Gable', 'Exterior1st_BrkFace', 'Exterior1st_HdBoard', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior2nd_VinylSd', 'MasVnrType_BrkFace', 'ExterQual_Gd', 'ExterQual_TA', 'ExterCond_Gd', 'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'BsmtQual_Ex', 'BsmtQual_Fa', 'BsmtCond_Fa', 'BsmtExposure_Gd', 'BsmtFinType1_ALQ', 'HeatingQC_Ex', 'HeatingQC_TA', 'CentralAir_N', 'Electrical_FuseA', 'KitchenQual_Ex', 'KitchenQual_TA', 'Functional_Typ', 'FireplaceQu_Gd', 'FireplaceQu_TA', 'GarageType_Attchd', 'GarageType_Detchd', 'GarageFinish_RFn', 'GarageFinish_Unf', 'PavedDrive_Y', 'Fence_GdWo', 'Fence_NA', 'MoSold_1', 'MoSold_10', 'MoSold_3', 'MoSold_5', 'MoSold_7', 'MoSold_9', 'YrSold_2006', 'YrSold_2007', 'YrSold_2009', 'YrSold_2010', 'SaleType_New', 'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_Normal']]
#print('X', X.shape, 'y', y.shape, 'X_test', X_test.shape)
#0.120 kaggle

#lightgbm cv = 20
#X =X[['Id', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'MiscVal', 'HouseYear', 'YrBltAndRemod', 'TotalSF', 'AllSF', 'AllBathrooms', 'PorchSF', 'Overallscore', 'ExtrenalScore', 'BSMTScore', 'GarageScore', 'AllKitchenScore', 'FP_heat_FenceScore', 'AllScore', 'MSSubClass_20', 'MSSubClass_30', 'MSSubClass_70', 'MSSubClass_80', 'MSSubClass_90', 'MSZoning_RL', 'MSZoning_RM', 'Alley_NA', 'LotShape_IR1', 'LotShape_Reg', 'LandContour_Bnk', 'LotConfig_Corner', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LandSlope_Gtl', 'LandSlope_Mod', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_IDOTRR', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NoRidge', 'Neighborhood_OldTown', 'Neighborhood_Sawyer', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Condition1_Artery', 'Condition1_Norm', 'BldgType_1Fam', 'HouseStyle_1.5Fin', 'HouseStyle_2Story', 'HouseStyle_SLvl', 'RoofStyle_Gable', 'Exterior1st_BrkFace', 'Exterior1st_HdBoard', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior2nd_HdBoard', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'MasVnrType_BrkFace', 'MasVnrType_Stone', 'ExterQual_Gd', 'ExterQual_TA', 'ExterCond_Gd', 'ExterCond_TA', 'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'BsmtQual_Ex', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtCond_Fa', 'BsmtExposure_Av', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_ALQ', 'BsmtFinType1_Rec', 'BsmtFinType2_Rec', 'Heating_GasA', 'HeatingQC_Ex', 'HeatingQC_TA', 'CentralAir_N', 'Electrical_FuseA', 'KitchenQual_Ex', 'KitchenQual_Gd', 'KitchenQual_TA', 'Functional_Typ', 'FireplaceQu_Gd', 'FireplaceQu_TA', 'GarageType_Attchd', 'GarageType_Detchd', 'GarageFinish_Fin', 'GarageFinish_RFn', 'GarageFinish_Unf', 'GarageQual_Fa', 'PavedDrive_Y', 'Fence_GdWo', 'Fence_NA', 'MoSold_1', 'MoSold_10', 'MoSold_11', 'MoSold_3', 'MoSold_4', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'MoSold_9', 'YrSold_2006', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', 'SaleType_New', 'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_Normal']]
#X_test =X_test[['Id', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'MiscVal', 'HouseYear', 'YrBltAndRemod', 'TotalSF', 'AllSF', 'AllBathrooms', 'PorchSF', 'Overallscore', 'ExtrenalScore', 'BSMTScore', 'GarageScore', 'AllKitchenScore', 'FP_heat_FenceScore', 'AllScore', 'MSSubClass_20', 'MSSubClass_30', 'MSSubClass_70', 'MSSubClass_80', 'MSSubClass_90', 'MSZoning_RL', 'MSZoning_RM', 'Alley_NA', 'LotShape_IR1', 'LotShape_Reg', 'LandContour_Bnk', 'LotConfig_Corner', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LandSlope_Gtl', 'LandSlope_Mod', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_IDOTRR', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NoRidge', 'Neighborhood_OldTown', 'Neighborhood_Sawyer', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Condition1_Artery', 'Condition1_Norm', 'BldgType_1Fam', 'HouseStyle_1.5Fin', 'HouseStyle_2Story', 'HouseStyle_SLvl', 'RoofStyle_Gable', 'Exterior1st_BrkFace', 'Exterior1st_HdBoard', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior2nd_HdBoard', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'MasVnrType_BrkFace', 'MasVnrType_Stone', 'ExterQual_Gd', 'ExterQual_TA', 'ExterCond_Gd', 'ExterCond_TA', 'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'BsmtQual_Ex', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtCond_Fa', 'BsmtExposure_Av', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_ALQ', 'BsmtFinType1_Rec', 'BsmtFinType2_Rec', 'Heating_GasA', 'HeatingQC_Ex', 'HeatingQC_TA', 'CentralAir_N', 'Electrical_FuseA', 'KitchenQual_Ex', 'KitchenQual_Gd', 'KitchenQual_TA', 'Functional_Typ', 'FireplaceQu_Gd', 'FireplaceQu_TA', 'GarageType_Attchd', 'GarageType_Detchd', 'GarageFinish_Fin', 'GarageFinish_RFn', 'GarageFinish_Unf', 'GarageQual_Fa', 'PavedDrive_Y', 'Fence_GdWo', 'Fence_NA', 'MoSold_1', 'MoSold_10', 'MoSold_11', 'MoSold_3', 'MoSold_4', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'MoSold_9', 'YrSold_2006', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', 'SaleType_New', 'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_Normal']]
#0.120 kaggle 

#Ridge cv = 20
#X = X[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'ScreenPorch', 'PoolArea', 'MiscVal', 'HavePool', 'Have2ndfloor', 'HaveGarage', 'HaveBSMT', 'HaveFireP', 'HouseYear', 'TotalSF', 'AllSF', 'AllBathrooms', 'PorchSF', 'Overallscore', 'BSMTScore', 'GarageScore', 'AllKitchenScore', 'FP_heat_FenceScore', 'AllScore', 'MSSubClass_120', 'MSSubClass_160', 'MSSubClass_180', 'MSSubClass_190', 'MSSubClass_20', 'MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45', 'MSSubClass_50', 'MSSubClass_60', 'MSSubClass_70', 'MSSubClass_75', 'MSSubClass_80', 'MSSubClass_85', 'MSSubClass_90', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'Alley_Grvl', 'Alley_NA', 'Alley_Pave', 'LotShape_IR1', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LotConfig_Corner', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside', 'LandSlope_Gtl', 'LandSlope_Mod', 'LandSlope_Sev', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition1_Artery', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'Condition2_Artery', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_RRNn', 'BldgType_1Fam', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'RoofStyle_Flat', 'RoofStyle_Gable', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'RoofMatl_CompShg', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'Exterior1st_AsbShng', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior1st_WdShing', 'Exterior2nd_AsbShng', 'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn', 'Exterior2nd_CmentBd', 'Exterior2nd_HdBoard', 'Exterior2nd_ImStucc', 'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng', 'MasVnrType_BrkCmn', 'MasVnrType_BrkFace', 'MasVnrType_NA', 'MasVnrType_None', 'MasVnrType_Stone', 'ExterQual_Ex', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterCond_Ex', 'ExterCond_Fa', 'ExterCond_Gd', 'ExterCond_TA', 'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'BsmtQual_Ex', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_NA', 'BsmtQual_TA', 'BsmtCond_Fa', 'BsmtCond_Gd', 'BsmtCond_NA', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Av', 'BsmtExposure_Gd', 'BsmtExposure_Mn', 'BsmtExposure_NA', 'BsmtExposure_No', 'BsmtFinType1_ALQ', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_NA', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_ALQ', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_NA', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'HeatingQC_Ex', 'HeatingQC_Fa', 'HeatingQC_TA', 'CentralAir_N', 'CentralAir_Y', 'Electrical_FuseA', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_SBrkr', 'KitchenQual_Ex', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'Functional_Maj1', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Typ', 'FireplaceQu_Ex', 'FireplaceQu_Fa', 'FireplaceQu_Gd', 'FireplaceQu_NA', 'FireplaceQu_Po', 'GarageType_2Types', 'GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'GarageType_NA', 'GarageFinish_Fin', 'GarageFinish_NA', 'GarageFinish_RFn', 'GarageFinish_Unf', 'GarageQual_Ex', 'GarageQual_Fa', 'GarageQual_Gd', 'GarageQual_NA', 'GarageQual_Po', 'GarageQual_TA', 'GarageCond_Ex', 'GarageCond_Fa', 'GarageCond_Gd', 'GarageCond_NA', 'GarageCond_Po', 'GarageCond_TA', 'PavedDrive_N', 'PavedDrive_P', 'PavedDrive_Y', 'Fence_GdPrv', 'Fence_GdWo', 'Fence_MnPrv', 'Fence_MnWw', 'Fence_NA', 'MiscFeature_Gar2', 'MiscFeature_NA', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MoSold_1', 'MoSold_10', 'MoSold_11', 'MoSold_12', 'MoSold_2', 'MoSold_4', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'MoSold_8', 'MoSold_9', 'YrSold_2006', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial']]
#X_test = X_test[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'ScreenPorch', 'PoolArea', 'MiscVal', 'HavePool', 'Have2ndfloor', 'HaveGarage', 'HaveBSMT', 'HaveFireP', 'HouseYear', 'TotalSF', 'AllSF', 'AllBathrooms', 'PorchSF', 'Overallscore', 'BSMTScore', 'GarageScore', 'AllKitchenScore', 'FP_heat_FenceScore', 'AllScore', 'MSSubClass_120', 'MSSubClass_160', 'MSSubClass_180', 'MSSubClass_190', 'MSSubClass_20', 'MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45', 'MSSubClass_50', 'MSSubClass_60', 'MSSubClass_70', 'MSSubClass_75', 'MSSubClass_80', 'MSSubClass_85', 'MSSubClass_90', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'Alley_Grvl', 'Alley_NA', 'Alley_Pave', 'LotShape_IR1', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LotConfig_Corner', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside', 'LandSlope_Gtl', 'LandSlope_Mod', 'LandSlope_Sev', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition1_Artery', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'Condition2_Artery', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_RRNn', 'BldgType_1Fam', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'RoofStyle_Flat', 'RoofStyle_Gable', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'RoofMatl_CompShg', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'Exterior1st_AsbShng', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior1st_WdShing', 'Exterior2nd_AsbShng', 'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn', 'Exterior2nd_CmentBd', 'Exterior2nd_HdBoard', 'Exterior2nd_ImStucc', 'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng', 'MasVnrType_BrkCmn', 'MasVnrType_BrkFace', 'MasVnrType_NA', 'MasVnrType_None', 'MasVnrType_Stone', 'ExterQual_Ex', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterCond_Ex', 'ExterCond_Fa', 'ExterCond_Gd', 'ExterCond_TA', 'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'BsmtQual_Ex', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_NA', 'BsmtQual_TA', 'BsmtCond_Fa', 'BsmtCond_Gd', 'BsmtCond_NA', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Av', 'BsmtExposure_Gd', 'BsmtExposure_Mn', 'BsmtExposure_NA', 'BsmtExposure_No', 'BsmtFinType1_ALQ', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_NA', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_ALQ', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_NA', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'HeatingQC_Ex', 'HeatingQC_Fa', 'HeatingQC_TA', 'CentralAir_N', 'CentralAir_Y', 'Electrical_FuseA', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_SBrkr', 'KitchenQual_Ex', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'Functional_Maj1', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Typ', 'FireplaceQu_Ex', 'FireplaceQu_Fa', 'FireplaceQu_Gd', 'FireplaceQu_NA', 'FireplaceQu_Po', 'GarageType_2Types', 'GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'GarageType_NA', 'GarageFinish_Fin', 'GarageFinish_NA', 'GarageFinish_RFn', 'GarageFinish_Unf', 'GarageQual_Ex', 'GarageQual_Fa', 'GarageQual_Gd', 'GarageQual_NA', 'GarageQual_Po', 'GarageQual_TA', 'GarageCond_Ex', 'GarageCond_Fa', 'GarageCond_Gd', 'GarageCond_NA', 'GarageCond_Po', 'GarageCond_TA', 'PavedDrive_N', 'PavedDrive_P', 'PavedDrive_Y', 'Fence_GdPrv', 'Fence_GdWo', 'Fence_MnPrv', 'Fence_MnWw', 'Fence_NA', 'MiscFeature_Gar2', 'MiscFeature_NA', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MoSold_1', 'MoSold_10', 'MoSold_11', 'MoSold_12', 'MoSold_2', 'MoSold_4', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'MoSold_8', 'MoSold_9', 'YrSold_2006', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial']]

#Ridge cv = 10
#X =X[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'HavePool', 'Have2ndfloor', 'HaveGarage', 'HaveBSMT', 'HaveFireP', 'HouseYear', 'YrBltAndRemod', 'TotalSF', 'AllSF', 'AllBathrooms', 'PorchSF', 'Overallscore', 'ExtrenalScore', 'BSMTScore', 'GarageScore', 'AllKitchenScore', 'FP_heat_FenceScore', 'AllScore', 'MSSubClass_120', 'MSSubClass_160', 'MSSubClass_180', 'MSSubClass_190', 'MSSubClass_20', 'MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45', 'MSSubClass_50', 'MSSubClass_60', 'MSSubClass_70', 'MSSubClass_75', 'MSSubClass_80', 'MSSubClass_85', 'MSSubClass_90', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'Alley_Grvl', 'Alley_NA', 'Alley_Pave', 'LotShape_IR1', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LotConfig_Corner', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside', 'LandSlope_Gtl', 'LandSlope_Mod', 'LandSlope_Sev', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition1_Artery', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'Condition2_Artery', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_RRNn', 'BldgType_1Fam', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'RoofStyle_Flat', 'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'RoofMatl_CompShg', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'Exterior1st_AsbShng', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior1st_WdShing', 'Exterior2nd_AsbShng', 'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn', 'Exterior2nd_BrkFace', 'Exterior2nd_CmentBd', 'Exterior2nd_HdBoard', 'Exterior2nd_ImStucc', 'Exterior2nd_MetalSd', 'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng', 'MasVnrType_BrkCmn', 'MasVnrType_BrkFace', 'MasVnrType_NA', 'MasVnrType_None', 'MasVnrType_Stone', 'ExterQual_Ex', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'ExterCond_Ex', 'ExterCond_Fa', 'ExterCond_Gd', 'ExterCond_TA', 'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'BsmtQual_Ex', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_NA', 'BsmtQual_TA', 'BsmtCond_Fa', 'BsmtCond_Gd', 'BsmtCond_NA', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Av', 'BsmtExposure_Gd', 'BsmtExposure_Mn', 'BsmtExposure_NA', 'BsmtExposure_No', 'BsmtFinType1_ALQ', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_NA', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_ALQ', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_NA', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'HeatingQC_Ex', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'CentralAir_N', 'CentralAir_Y', 'Electrical_FuseA', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_SBrkr', 'KitchenQual_Ex', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'Functional_Maj1', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Typ', 'FireplaceQu_Ex', 'FireplaceQu_Fa', 'FireplaceQu_Gd', 'FireplaceQu_NA', 'FireplaceQu_Po', 'FireplaceQu_TA', 'GarageType_2Types', 'GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'GarageType_NA', 'GarageFinish_Fin', 'GarageFinish_NA', 'GarageFinish_RFn', 'GarageFinish_Unf', 'GarageQual_Ex', 'GarageQual_Fa', 'GarageQual_Gd', 'GarageQual_NA', 'GarageQual_Po', 'GarageQual_TA', 'GarageCond_Ex', 'GarageCond_Fa', 'GarageCond_Gd', 'GarageCond_NA', 'GarageCond_Po', 'GarageCond_TA', 'PavedDrive_N', 'PavedDrive_P', 'PavedDrive_Y', 'Fence_GdPrv', 'Fence_GdWo', 'Fence_MnPrv', 'Fence_MnWw', 'Fence_NA', 'MiscFeature_Gar2', 'MiscFeature_NA', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MoSold_1', 'MoSold_10', 'MoSold_11', 'MoSold_12', 'MoSold_2', 'MoSold_3', 'MoSold_4', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'MoSold_8', 'MoSold_9', 'YrSold_2006', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial']]
#X_test = X_test[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'HavePool', 'Have2ndfloor', 'HaveGarage', 'HaveBSMT', 'HaveFireP', 'HouseYear', 'YrBltAndRemod', 'TotalSF', 'AllSF', 'AllBathrooms', 'PorchSF', 'Overallscore', 'ExtrenalScore', 'BSMTScore', 'GarageScore', 'AllKitchenScore', 'FP_heat_FenceScore', 'AllScore', 'MSSubClass_120', 'MSSubClass_160', 'MSSubClass_180', 'MSSubClass_190', 'MSSubClass_20', 'MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45', 'MSSubClass_50', 'MSSubClass_60', 'MSSubClass_70', 'MSSubClass_75', 'MSSubClass_80', 'MSSubClass_85', 'MSSubClass_90', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'Alley_Grvl', 'Alley_NA', 'Alley_Pave', 'LotShape_IR1', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LotConfig_Corner', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside', 'LandSlope_Gtl', 'LandSlope_Mod', 'LandSlope_Sev', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition1_Artery', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'Condition2_Artery', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_RRNn', 'BldgType_1Fam', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'RoofStyle_Flat', 'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'RoofMatl_CompShg', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'Exterior1st_AsbShng', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior1st_WdShing', 'Exterior2nd_AsbShng', 'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn', 'Exterior2nd_BrkFace', 'Exterior2nd_CmentBd', 'Exterior2nd_HdBoard', 'Exterior2nd_ImStucc', 'Exterior2nd_MetalSd', 'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng', 'MasVnrType_BrkCmn', 'MasVnrType_BrkFace', 'MasVnrType_NA', 'MasVnrType_None', 'MasVnrType_Stone', 'ExterQual_Ex', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'ExterCond_Ex', 'ExterCond_Fa', 'ExterCond_Gd', 'ExterCond_TA', 'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'BsmtQual_Ex', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_NA', 'BsmtQual_TA', 'BsmtCond_Fa', 'BsmtCond_Gd', 'BsmtCond_NA', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Av', 'BsmtExposure_Gd', 'BsmtExposure_Mn', 'BsmtExposure_NA', 'BsmtExposure_No', 'BsmtFinType1_ALQ', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_NA', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_ALQ', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_NA', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'HeatingQC_Ex', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'CentralAir_N', 'CentralAir_Y', 'Electrical_FuseA', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_SBrkr', 'KitchenQual_Ex', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'Functional_Maj1', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Typ', 'FireplaceQu_Ex', 'FireplaceQu_Fa', 'FireplaceQu_Gd', 'FireplaceQu_NA', 'FireplaceQu_Po', 'FireplaceQu_TA', 'GarageType_2Types', 'GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'GarageType_NA', 'GarageFinish_Fin', 'GarageFinish_NA', 'GarageFinish_RFn', 'GarageFinish_Unf', 'GarageQual_Ex', 'GarageQual_Fa', 'GarageQual_Gd', 'GarageQual_NA', 'GarageQual_Po', 'GarageQual_TA', 'GarageCond_Ex', 'GarageCond_Fa', 'GarageCond_Gd', 'GarageCond_NA', 'GarageCond_Po', 'GarageCond_TA', 'PavedDrive_N', 'PavedDrive_P', 'PavedDrive_Y', 'Fence_GdPrv', 'Fence_GdWo', 'Fence_MnPrv', 'Fence_MnWw', 'Fence_NA', 'MiscFeature_Gar2', 'MiscFeature_NA', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MoSold_1', 'MoSold_10', 'MoSold_11', 'MoSold_12', 'MoSold_2', 'MoSold_3', 'MoSold_4', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'MoSold_8', 'MoSold_9', 'YrSold_2006', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial']]

#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
  

kfolds20 = KFold(n_splits=20, shuffle=True)
kfolds10 = KFold(n_splits=10, shuffle=True)
# rmse funct
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

print('Constructing Model')

# normal model
'''
estimator=RidgeCV(alphas=[0.0001,0.001,0.01,0.1,1,10,100])
parameter_grid={
        'fit_intercept':[False,True],
        'normalize':[False,True],
        'cv':range(2,10),'verbose' : [1]}
grid = RandomizedSearchCV(estimator,parameter_grid,cv = 3,scoring = 'neg_mean_squared_error',n_iter=1000,n_jobs = 3)
grid.fit(X, y)
print ('Ridge:',grid.best_params_)
ridge = RidgeCV(alphas=[0.0001,0.001,0.01,0.1,1,10,100])
ridge = ridge.set_params(**grid.best_params_)

best_params.append(grid.best_params_)
'''
ridge = make_pipeline(RobustScaler(), RidgeCV())

'''
estimator=KernelRidge()
parameters = {'alpha': uniform(0.05, 1.0), 'kernel': ['polynomial'], 
          'degree': [2], 'coef0':uniform(0.5, 3.5)}

grid = RandomizedSearchCV(estimator = estimator,
                               param_distributions = parameters,
                               n_iter = 1000,
                               cv = 3,
                               scoring ='neg_mean_squared_error',
                               n_jobs = 3)
grid.fit(X, y)
print ('KernelRidge:',grid.best_params_)
Keridge = KernelRidge()
Keridge = Keridge.set_params(**grid.best_params_)  
best_params.append(grid.best_params_)
'''    
Keridge = KernelRidge()

'''
estimator=LassoCV(alphas=[0.0001,0.001,0.01,0.1,1,10,100])
parameter_grid={
        'n_alphas':range(50,200),
        'fit_intercept':[False,True],
        'normalize':[False,True],
        'cv':range(2,10),
        'max_iter':range(500,2000,100)}
grid = RandomizedSearchCV(estimator,parameter_grid,cv = 3,scoring = 'neg_mean_squared_error',n_iter=1000,n_jobs = 3)
grid.fit(X, y)
print ('Lasso:',grid.best_params_)

lasso = LassoCV(alphas=[0.0001,0.001,0.01,0.1,1,10,100])
lasso = lasso.set_params(**grid.best_params_)

best_params.append(grid.best_params_)
'''
lasso = make_pipeline(RobustScaler(), LassoCV())

'''
estimator=ElasticNetCV(alphas=[0.0001,0.001,0.01,0.1,1,10,100], l1_ratio = [0.1, 0.5, 0.9, 0.95, 0.99, 1])
parameter_grid={
        'n_alphas':range(50,200),
        'fit_intercept':[False,True],
        'normalize':[False,True],
        'cv':range(2,10),
        'max_iter':range(500,2000,100)}
grid = RandomizedSearchCV(estimator,parameter_grid,cv = 3,scoring = 'neg_mean_squared_error',n_iter=1000,n_jobs = 3)
grid.fit(X, y)
print ('ElasticNet:',grid.best_params_)
elasticnet = ElasticNetCV(alphas=[0.0001,0.001,0.01,0.1,1,10,100], l1_ratio = [0.1, 0.5, 0.9, 0.95, 0.99, 1])
elasticnet = elasticnet.set_params(**grid.best_params_)
best_params.append(grid.best_params_) 
'''    
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV())    

'''
estimator=SVR()
parameter_grid={
        'degree':range(1,10),
        'gamma':np.arange(0.0000,0.5,0.0001),
        'C':np.arange(0.1,3,0.2),
        'epsilon':np.arange(0.000,0.5,0.001)}
grid = RandomizedSearchCV(estimator,parameter_grid,cv = 3,scoring = 'neg_mean_squared_error',n_iter=1000,n_jobs = 3)
grid.fit(X, y)
print ('SVR:',grid.best_params_)         
svr = SVR()
svr = svr.set_params(**grid.best_params_)  
best_params.append(grid.best_params_)
'''                                 
svr = make_pipeline(RobustScaler(), SVR())

'''
estimator=GradientBoostingRegressor()
parameter_grid={
        'learning_rate':np.arange(0.05,0.25,0.01),
        'n_estimators':range(100,3000,100),
        'max_depth':range(10,30),
        'min_samples_leaf':range(70,90),
        'verbose' : [1]}
grid = RandomizedSearchCV(estimator,parameter_grid,cv = 3,scoring = 'neg_mean_squared_error',n_iter=100,n_jobs = 3)
grid.fit(X, y)
print ('GBR:',grid.best_params_)
gbr = GradientBoostingRegressor()
gbr = gbr.set_params(**grid.best_params_) 

best_params.append(grid.best_params_)
'''
gbr = GradientBoostingRegressor()

'''
estimator=LGBMRegressor()
parameter_grid={
        'max_depth':range(2,5,1),
        'learning_rate':np.linspace(0.001,1,20),
        'feature_fraction':np.linspace(0.5,0.99,20),
        'bagging_fraction':np.linspace(0.1,0.99,20),
        'bagging_frequency':range(5,10,1),
        'num_leaves':range(100,200,5),
        'min_data_in_leaf':range(50,200,10),
        'n_estimators':range(100,5000,100),
        'verbose' : [1]}
grid = RandomizedSearchCV(estimator,parameter_grid,cv = 3,scoring = 'neg_mean_squared_error',n_iter=1000,n_jobs = 3)
grid.fit(X, y)
print ('lightgbm:',grid.best_params_)       
lightgbm = LGBMRegressor()
lightgbm = lightgbm.set_params(**grid.best_params_)  
best_params.append(grid.best_params_)
'''
lightgbm = LGBMRegressor()


xgbRF = xg.XGBRFRegressor()

'''              
estimator=XGBRegressor()
parameter_grid={
        'max_depth':range(2,5,1),
        'learning_rate':np.linspace(0.001,1,20),
        'colsample_bytree':np.linspace(0.1,0.99,20),
        'num_leaves':range(100,200,5),
        'n_estimators':range(100,5000,100),
        'verbose' : [1]}
grid = RandomizedSearchCV(estimator,parameter_grid,cv = 3,scoring = 'neg_root_mean_squared_error',n_iter=1000,n_jobs = 3)
grid.fit(X, y)
print ('Xgb:',grid.best_params_)              
xgboost = XGBRegressor()
xgboost = xgboost.set_params(**grid.best_params_)  
best_params.append(grid.best_params_)
'''
xgboost = XGBRegressor()

#Voting model 
voting_gen = VotingRegressor(estimators=[('ridge',ridge),('lasso',lasso),('elasticnet',elasticnet),
                                              ('lightgbm',lightgbm),('Keridge',Keridge)])                                 
#Stacking model
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, voting_gen,Keridge),
                                meta_regressor=lightgbm, use_features_in_secondary=True)
print('Construct Model Finished')
'''
selector = RFECV(elasticnet,step=1, cv =20,n_jobs =-1,scoring = 'neg_root_mean_squared_error')
selector = selector.fit(X,y)
test_score = selector.score(X,y)
R2Score = r2_score(selector.predict(X),y)

print(' The number of selected features {:.4f} Testing accura0cy: {:4f} R2Score: {:4f}'.format(selector.n_features_, test_score,R2Score))
features = [f for f,s in zip(X.columns, selector.support_) if s]
print ('The selected features are : {}'.format(features))
temp_data = X[features]
csv_name = 'testing_RFECV'+str(i)+'.csv'
temp_data.to_csv(csv_name)
exit()
'''

print('-'*100)
print('Calculating 10 folds and 20 folds score')

score10 = np.sqrt(-cross_val_score(ridge, X, y, scoring="neg_mean_squared_error", cv=kfolds10))
score20 = np.sqrt(-cross_val_score(ridge, X, y, scoring="neg_mean_squared_error", cv=kfolds20))
print("Kernel Ridge score    : {:.4f} ({:.4f}) {:.4f} ({:.4f}) ".format(score10.mean(), score10.std(),score20.mean(),score20.std()))
score10 = np.sqrt(-cross_val_score(Keridge, X, y, scoring="neg_mean_squared_error", cv=kfolds10))
score20 = np.sqrt(-cross_val_score(Keridge, X, y, scoring="neg_mean_squared_error", cv=kfolds20))
print("Keridge score         : {:.4f} ({:.4f}) {:.4f} ({:.4f}) ".format(score10.mean(), score10.std(),score20.mean(),score20.std()))
score10 = np.sqrt(-cross_val_score(lasso, X, y, scoring="neg_mean_squared_error", cv=kfolds10))
score20 = np.sqrt(-cross_val_score(lasso, X, y, scoring="neg_mean_squared_error", cv=kfolds20))
print("Lasso score           : {:.4f} ({:.4f}) {:.4f} ({:.4f}) ".format(score10.mean(), score10.std(),score20.mean(),score20.std()))

score10 = np.sqrt(-cross_val_score(elasticnet, X, y, scoring="neg_mean_squared_error", cv=kfolds10))
score20 = np.sqrt(-cross_val_score(elasticnet, X, y, scoring="neg_mean_squared_error", cv=kfolds20))
print("ElasticNet score      : {:.4f} ({:.4f}) {:.4f} ({:.4f}) ".format(score10.mean(), score10.std(),score20.mean(),score20.std()))
score10 = np.sqrt(-cross_val_score(svr, X, y, scoring="neg_mean_squared_error", cv=kfolds10))
score20 = np.sqrt(-cross_val_score(svr, X, y, scoring="neg_mean_squared_error", cv=kfolds20))
print("SVR score             : {:.4f} ({:.4f}) {:.4f} ({:.4f}) ".format(score10.mean(), score10.std(),score20.mean(),score20.std()))
score10 = np.sqrt(-cross_val_score(lightgbm, X, y, scoring="neg_mean_squared_error", cv=kfolds10))
score20 = np.sqrt(-cross_val_score(lightgbm, X, y, scoring="neg_mean_squared_error", cv=kfolds20))
print("Lightgbm score        : {:.4f} ({:.4f}) {:.4f} ({:.4f}) ".format(score10.mean(), score10.std(),score20.mean(),score20.std()))
score10 = np.sqrt(-cross_val_score(gbr, X, y, scoring="neg_mean_squared_error", cv=kfolds10))
score20 = np.sqrt(-cross_val_score(gbr, X, y, scoring="neg_mean_squared_error", cv=kfolds20))
print("GradientBoosting score: {:.4f} ({:.4f}) {:.4f} ({:.4f}) ".format(score10.mean(), score10.std(),score20.mean(),score20.std()))

score10 = np.sqrt(-cross_val_score(xgboost, X, y, scoring="neg_mean_squared_error", cv=kfolds10))
score20 = np.sqrt(-cross_val_score(xgboost, X, y, scoring="neg_mean_squared_error", cv=kfolds20))
print("Xgboost score         : {:.4f} ({:.4f}) {:.4f} ({:.4f})".format(score10.mean(), score10.std(),score20.mean(),score20.std()))
score10 = np.sqrt(-cross_val_score(xgbRF, X, y, scoring="neg_mean_squared_error", cv=kfolds10))
score20 = np.sqrt(-cross_val_score(xgbRF, X, y, scoring="neg_mean_squared_error", cv=kfolds20))
print("xgbRF score           : {:.4f} ({:.4f}) {:.4f} ({:.4f}) ".format(score10.mean(), score10.std(),score20.mean(),score20.std()))

score10 = np.sqrt(-cross_val_score(voting_gen, X, y, scoring="neg_mean_squared_error", cv=kfolds10))
score20 = np.sqrt(-cross_val_score(voting_gen, X, y, scoring="neg_mean_squared_error", cv=kfolds20))
print("voting_gen score      : {:.4f} ({:.4f}) {:.4f} ({:.4f}) ".format(score10.mean(), score10.std(),score20.mean(),score20.std()))
score10 = np.sqrt(-cross_val_score(stack_gen, X, y, scoring="neg_mean_squared_error", cv=kfolds10))
score20 = np.sqrt(-cross_val_score(stack_gen, X, y, scoring="neg_mean_squared_error", cv=kfolds20))
print("stack_gen score       : {:.4f} ({:.4f}) {:.4f} ({:.4f}) ".format(score10.mean(), score10.std(),score20.mean(),score20.std()))

print('-'*100)
print('Model fiting')
print( 'ridge')
ridge = ridge.fit(X, y)
print('Keridge')
ker = Keridge.fit(X, y)
print('lasso')
lasso = lasso.fit(X, y)
print('elasticnet')
elastic = elasticnet.fit(X, y)
print('svr')
svr = svr.fit(X, y)
print('GradientBoosting')
gbr = gbr.fit(X, y)
print('lightgbm')
lgb = lightgbm.fit(X, y)
print('xgboost')
xgb = xgboost.fit(X, y)
print('xgbRF')
xgbRF = xgbRF.fit(X, y)
print('voting_gen')
voting_gen = voting_gen.fit(X, y)
print('StackingCVRegressor')
stack_gen = stack_gen.fit(np.array(X), np.array(y))

def blend_models_predict(X):
    return ((0.0 * elastic.predict(X)) + \
            (0.0 * lasso.predict(X)) + \
            (0.0* ridge.predict(X)) + \
            (0.0 * svr.predict(X)) + \
            (0.00 * gbr.predict(X)) + \
            (0.1 * lgb.predict(X)) + \
            (0.05 * ker.predict(X))+\
            (0.45 * voting_gen.predict(X))+\
            (0.40 * stack_gen.predict(X)))    

print('-'*100)
print('Create Solution')
print('-'*100)           
print('Trainging RMSE score :', rmse(y, blend_models_predict(X)))
print('Predict submission')
submission = pd.read_csv("sample_submission.csv")
#chnage the prediction at the predict submission file
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_test)))

submission.to_csv("House.csv", index=False)
print('Save submission')