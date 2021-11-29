# Import necessary modules


import pandas as pd
import numpy as np
import seaborn as sns


# ** Read in the Ecommerce Customers csv file as a DataFrame called customers.**



customers = pd.read_csv("Ecommerce Customers")


# **Check the head of customers, and check out its info() and describe() methods.**

customers.head()


customers.info()

customers.describe()


# ## Exploratory Data Analysis
# **Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**

sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=customers)


# ** Do the same but with the Time on App column instead. **

sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=customers)


# ** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

sns.jointplot(x="Time on Website", y="Yearly Amount Spent", kind="hex",data=customers)


# **Let's explore these types of relationships across the entire data set. Use [pairplot](https://stanford.edu/~mwaskom/software/seaborn/tutorial/axis_grids.html#plotting-pairwise-relationships-with-pairgrid-and-pairplot) to recreate the plot below.(Don't worry about the the colors)**

sns.pairplot(data=customers)


# **Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**
# Length of membership 


# **Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. **

sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=customers)

# ## Training and Testing Data
# ** Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. **

from sklearn.model_selection import train_test_split
X = customers[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]
y = customers["Yearly Amount Spent"]


# ** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



# ## Training the Model
# 
# Now its time to train our model on our training data!

from sklearn.linear_model import LinearRegression


# **Create an instance of a LinearRegression() model named lm.**

lm= LinearRegression() 


# ** Train/fit lm on the training data.**

lm.fit(X_train, y_train)
print(lm.coef_)


# **Print out the coefficients of the model**

print(lm.coef_)


# ## Predicting Test Data
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 
# ** Use lm.predict() to predict off the X_test set of the data.**

lm.predict(X_test)


# ** Create a scatterplot of the real test values versus the predicted values. **

predictions = lm.predict(X_test)
sns.scatterplot(y_test,predictions)


# ## Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

from sklearn import metrics
print("MAE:", metrics.mean_absolute_error(y_test, predictions))
print("MSE:", metrics.mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ## Residuals
# You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 
# **Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**


sns.distplot((y_test-predictions), bins=50)


# ## Conclusion
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.
# 
# Use the coefficients to determine where the company should put more effort into their comapny. Which area is giving the most return for their time?
df = pd.DataFrame(lm.coef_, X.columns)
df.columns = ["Coefficients"]
df

