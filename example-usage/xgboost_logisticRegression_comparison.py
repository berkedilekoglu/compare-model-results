from sklearn.datasets import make_moons #Library to create binary classification data
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from comparison_lib.comparison import *
# %matplotlib inline  THIS IS A MUST FOR .ipynb projects
get_ipython().magic('matplotlib inline')





X, y = make_moons(n_samples=20000, noise=0.1)





X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.40, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_test,y_test, test_size=0.5, random_state=42, stratify=y_test)



xg_model = XGBClassifier()
xg_model.fit(X_train,y_train)


# ## Before Prediction Create comparison class and sort True and False labels ##


comparison_device = comparison(X_val,y_val) #Create our class instance



Xx_val,yy_val = comparison_device.order_test_samples() #It is important to sort true and false labels in val data
#The aim of that part is just taking good illustration



yy_pred = xg_model.predict_proba(Xx_val)


# ## Set Probs for Our Model ##




comparison_device.set_prob_predictions("XGBoostModel",yy_pred,threshold=0.5)





comparison_device.set_prob_predictions("XGBoostModel2",yy_pred,threshold=0.85) #Lets also change threshold


# ## Same for LR ##



lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)


# #### Now our Gold Labels already sorted so skip this part! ####


yy_pred2 = lr_model.predict_proba(Xx_val)





comparison_device.set_prob_predictions("LRModel",yy_pred2,threshold=0.5)





comparison_device.set_prob_predictions("LRModel2",yy_pred2,threshold=0.85) #Lets also change threshold


# ## Let's Plot ##



comparison_device.plot_predictions()


# ### Let's take Comparison Report ###




comparison_device.compare_predictions(modelName1="XGBoostModel",modelName2="LRModel",modelName3="LRModel2")


# ## comparison class supports predicted labels directly ##




comparison_device.clear_all() ##Lets clear appended models





comparison_device = comparison(X_val,y_val) #Create our class instance





Xx_val,yy_val = comparison_device.order_test_samples() #It is important to sort true and false labels in val data
#The aim of that part is just taking good illustration





yy_pred = xg_model.predict(Xx_val)





comparison_device.set_label_predictions("XGBoost Model",yy_pred) #This time we will use set_label_predictions





yy_pred2 = lr_model.predict(Xx_val)





comparison_device.set_label_predictions("LR Model",yy_pred2)





comparison_device.plot_predictions()

