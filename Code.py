
# coding: utf-8

# In[83]:

#import the pandas module
import pandas as pd
ak=pd.read_csv("Data.csv") #Read function is used to read csv file


# In[84]:

#iloc function is used to show all the data of csv file
ak.iloc[:,:]


# In[85]:

#Drop the rows with Nan values
ak2 = ak.dropna(subset = ['Shipping_Type', 'No_of_customers_who_bought_the_product'])

#Show the Updated Table
ak2


# In[86]:

#show the data types of all the columns
ak2.dtypes


# In[87]:

#Describe Function is used to show the mean, std, min, max and various values
ak2.describe()


# In[88]:

#import matplotlib to show the graphs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().magic('matplotlib inline')


# In[89]:

L = ak2.groupby("Product_Name")["Geographic_Region" , "Sales_M"].mean()
L


# In[90]:

L.plot(y=["Sales_M",], kind="bar")
plt.ylabel("Sales_M")


# In[91]:

ak3 = ak2[['Product_Name' ,'Sales_M' ,'Year']]
ak3.iloc[:,:]


# In[92]:

#import matplotlib to show the graphs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().magic('matplotlib inline')


# In[93]:

style.use('classic')

S= ak3.iloc[0:189,:]
T= ak3.iloc[190:376,:]
P= ak3.iloc[377:568,:]
I= ak3.iloc[569:744,:]
A= ak3.iloc[745:955,:]

S= plt.scatter(S.Year,S.Sales_M,color="Green", label="Television")
T= plt.scatter(T.Year,T.Sales_M,color="Violet", label="Refrigerator")
P= plt.scatter(P.Year,P.Sales_M,color="Red", label="Washing Machine")
I= plt.scatter(I.Year,I.Sales_M,color="Orange", label="Vacuum Cleaner")
A= plt.scatter(A.Year,A.Sales_M,color="Black", label="Air Conditioner")

#set the x-axis label
plt.xlabel("Year")
#set the y-axis label
plt.ylabel("Monthly Sales($M)")
#set the title of the graph
plt.title("Change in Number of sales of different products Over the period 1818 to 2017")
plt.grid()

plt.legend(bbox_to_anchor=(0, 0, 1, 1), loc=2, borderaxespad=0.)


# In[94]:

ak2


# In[95]:

ak4 = ak2[['Product_Name' ,'Geographic_Region','Sales_M']]
ak4.iloc[:,:]


# In[96]:

p = ak4.groupby("Product_Name")["Sales_M"].mean()


# In[97]:

p


# In[98]:

import matplotlib.patches as mpatches

p = ak4.groupby(('Product_Name', 'Geographic_Region'))
means = p.mean()
means.plot.bar(color = 'rrrrbbbbccccggggyyyy')


plt.grid()
red_patch = mpatches.Patch(color='red', label='Air Conditioner')
blue_patch = mpatches.Patch(color='blue', label='Refrigerator')
cyan_patch = mpatches.Patch(color='cyan', label='Television')
green_patch = mpatches.Patch(color='green', label='Vacuum Cleaner')
yellow_patch = mpatches.Patch(color='yellow', label='Washing Machine')
plt.legend(handles=[red_patch, blue_patch, cyan_patch, green_patch, yellow_patch])


#set the y-axis label
plt.ylabel("Number of Sales ($M) ")
#set the title of the graph
plt.title("Number of sales in Each region for every book over the period 1818 to 2017")


# In[99]:

p.mean()


# In[100]:

#Loading the necessary files to display Naive Bayes
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
data.target_names


# In[101]:

#Defining all categories
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale',
 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

#Training the data on these categories
train = fetch_20newsgroups(subset='train', categories=categories)

#Testing the data for these categories
test = fetch_20newsgroups(subset='test', categories=categories)

#printing training data
print(train.data[5])


# In[102]:

print(test.data[5])


# In[103]:

print(len(train.data))


# In[104]:

#Import necessary packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Creating a model based on Multinomial Naive bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Training the model with the train data
model.fit(train.data, train.target)

#Creating lables for the test data
labels = model.predict(test.data)


# In[105]:

# Creating confusion matrix with heat map
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
          , xticklabels=train.target_names,
           yticklabels=train.target_names)

#Plotting heatmap of confusion matrix
#set the y-axis label
plt.xlabel("True Label")
#set the title of the graph
plt.ylabel("Predicted Label")


# In[106]:

#Predicting category on new data beased on trained model
def predict_category(s, train=train, model=model):
    pred=model.predict([s])
    return train.target_names[pred[0]]


# In[ ]:



