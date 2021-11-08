
# Multilinear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data preprocessing

#lets load the data set
aprice = pd.read_csv("G:\\DS assignments\\Multi linear Reg\\Datasets_MLR\\Avacado_Price.csv")
aprice.describe()

#to creat the dummies
aprice['region'].nunique()
aprice['type'].nunique()

#lets creat the dummies for type variable

aprice = pd.get_dummies(aprice.drop(['region'],axis=1),drop_first=True)
aprice.head()
aprice.tail()
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#AveragePrice

plt.bar(height = aprice.AveragePrice, x = np.arange(1, 18250, 1))
plt.hist(aprice.AveragePrice) #histogram
plt.boxplot(aprice.AveragePrice) #boxplot

#correlation plot
import seaborn as sns
plt.figure(figsize=(12,6))
sns.heatmap(aprice.corr(),cmap='coolwarm',annot=True)

# Jointplot
import seaborn as sns
sns.jointplot(x=aprice['Total_Volume'], y=aprice['AveragePrice'])
sns.jointplot(x=aprice['Total_Bags'], y=aprice['AveragePrice'])


# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(aprice['Total_Volume'])
sns.countplot(aprice['Total_Bags'])


# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(aprice.Total_Volume, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(aprice.iloc[:, :])
                             
# Correlation matrix 
aprice.corr()


# preparing model considering all the variables 
 # preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
aprice.columns  
     
ml1 = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3+Total_Bags+Small_Bags+Large_Bags+year+type_organic', data = aprice).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)


aprice_new = aprice.drop(aprice.index[[80]])

# Preparing model                  
ml_new = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3+Total_Bags+Small_Bags+Large_Bags+year+type_organic', data =aprice_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_ap = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3+Total_Bags+Small_Bags+Large_Bags+year+type_organic', data = aprice_new).fit().rsquared  
vif_ap = 1/(1 - rsq_ap) 

rsq_tv = smf.ols('Total_Volume ~ AveragePrice + tot_ava1 + tot_ava2 + tot_ava3+Total_Bags+Small_Bags+Large_Bags+year+type_organic', data = aprice_new).fit().rsquared  
vif_tv = 1/(1 - rsq_tv)

rsq_tb = smf.ols('Total_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3+AveragePrice+Small_Bags+Large_Bags+year+type_organic', data = aprice_new).fit().rsquared  
vif_tb = 1/(1 - rsq_tb) 

rsq_to = smf.ols('type_organic ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3+Total_Bags+Small_Bags+Large_Bags+year+AveragePrice', data = aprice_new).fit().rsquared  
vif_to = 1/(1 - rsq_to) 


# Storing vif values in a data frame
d1 = {'Variables':['AveragePrice', 'Total_Volume', 'Total_Bags', 'type_organic'], 'VIF':[vif_ap, vif_tv, vif_tb, vif_to]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As Total_Volume is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('AveragePrice ~ tot_ava1 + tot_ava2 + tot_ava3+Total_Bags+Small_Bags+Large_Bags+year+type_organic', data = aprice_new).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(aprice_new)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = aprice_new.AveragePrice, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
aprice_train, aprice_test = train_test_split(aprice, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3+Total_Bags+Small_Bags+Large_Bags+year+type_organic", data = aprice_train).fit()

# prediction on test data set 
test_pred = model_train.predict(aprice_test)

# test residual values 
test_resid = test_pred - aprice_test.AveragePrice
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(aprice_train)

# train residual values 
train_resid  = train_pred - aprice_train.AveragePrice
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
