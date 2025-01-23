# import libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# App heading
st.write("""
# Explore different ML Models and Datasets
Daikhty hain kon sa best ha in main se?""")

# Using dataset names in sidebar selectbox
dataset_name=st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine"))

# Using Classifier names in sidebar selectbox
classifier_name=st.sidebar.selectbox("Select classifier",("KNN","SVM","Random Forest"))

# Define a function to load dataset
def load_dataset(dataset_name):
    data=None
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name=="Wine":
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    x=data.data
    y=data.target
    return x,y
# Now use this function and equal it to X and y
X,y=load_dataset(dataset_name)

# Shape of dataset:
st.write("Shape of dataset:", X.shape)
st.write("Class of dataset:",len(np.unique(y)))

# Add parameters of different classifiers in user input
def add_parameter_ui(classifier_name):
    params=dict()
    if classifier_name=="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        params['C']=C #its the # import libraries
    elif classifier_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        params["K"]=K
    else:
        max_depth=st.sidebar.slider("max_depth",2,15)
        params["max_depth"]=max_depth                # depth of every tree that grows in random forest
        n_estimators=st.sidebar.slider("n_estimators",1,100)
        params["n_estimators"]=n_estimators
    return params
# call function
params=add_parameter_ui(classifier_name)

# Make classifier base on classifier_names and params
def get_classifier(classifier_name,params):
    clf=None
    if classifier_name=="SVM":
        clf=SVC(C=params['C'])
    elif classifier_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf=RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1234)
    return clf
# Now equal clf variable with function
clf=get_classifier(classifier_name ,params)

# Now we split data into train and test by 80/20 ratio
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2,random_state=1234)

# Now train classifier
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

# Now we will check accuracy score and print it on app
acc=accuracy_score(y_test,y_pred)
st.write(f"Classifier={classifier_name}")
st.write(f"Accuracy={acc:.2f}")

### PLOT DATASET ###
# Now we will draw all our features in two dimensional plot
pca=PCA(2)
X_projected=pca.fit_transform(X)

# Now we will slice our data in 0,1 dimension
x1=X_projected[:,0]
x2=X_projected[:,1]

# Create a proper figure
fig,ax=plt.subplots()
scatter=ax.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
fig.colorbar(scatter,ax=ax)

# plt.show()
st.pyplot(fig)




