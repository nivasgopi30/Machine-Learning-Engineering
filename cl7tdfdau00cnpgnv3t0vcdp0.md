## Introduction to Machine Learning

Hello all, welcome to my blog. This is the first blog of my Machine Learning Series. Every week, I will try to publish one blog on what I learned throughout that week. I'm learning MLEngineering from Datatalks Club's ML Zoomcamp 2022. In this blog, I will take you through some small concepts which will introduce Machine Learning.

When there are traditional software engineering principles, what is the requirement of Machine Learning? Let's take an example of spam mail classification. On what basis will you classify a mail whether it is spam or good? Through some common phishing words. Isn't it? Junk mail can contain many common words. Will you define hundreds of methods/functions to let the system know, if the mail contains this word, make it spam, or if the mail contains that word, make it as good? Too tedious work right!!?? What if there is a system which can learn itself based on experience and identify whether mail is spam or good? That's where Machine Learning comes into play. Machine Learning is of three types:-

**1) Supervised Learning (Classification, Regression)
2) Unsupervised Learning (Clustering, Anomaly Detection)
3) Reinforcement Learning (Recommendation Systems)**

We will just talk about Supervised. I will try to cover the rest in my upcoming articles. Mathematically, Supervised Learning is just one equation i.e g(X) = y. Every algorithm will best try to map a function "g" for the input "X" to give the output "y". This is meant by Prediction (nothing else). Easy Right??

For doing any ML task, we can follow a process called CRISP-DM. That is CRoss Industry Standard Process for Data Mining. Even though it is old, as proposed by IBM, it applies to many projects nowadays also. Our ML Project can be divided into 6 stages. They are:

1) Business Understanding
2) Data Understanding
3) Data Preparation
4) Modeling
5) Evaluation
6) Deployment

Let's see what are those.

**1) Business Understanding:-**

First of all, we should understand what the business problem is about. Should assess whether the problem requires ML, if not then we can solve it with traditional SWE principles.

**2) Data Understanding:-**

If the problem requires ML, collect relevant data. Check whether the data is enough or if more data is required. Also which type of data is required and where we can get the data.

**3) Data Preparation:-**

Prepare / Wrangle / Clean the raw data into a form that can be fitted to the machine learning model.

**4) Modeling:-**

We should experiment which model can be the right one for our task i.e whether it can be decision trees, random forests, neural networks etc.

**5) Evaluation:-**

Evaluate the performance of every model. Then, finalize the model which is having more performance. We can evaluate our model by testing it on 5-10% of our users. Then we can know whether it succeeded or not.

**6) Deployment:-**

Deploy the finalized model after evaluation into our web service. Also, make sure it is stable and maintainable (majorly this job is taken care of by MLOps engineers / Site Reliability Engineers ).

### Model Selection Process:-

How should we select an ML model for our task? It depends on your task. If you are not sure which will work, then you should experiment with different ones. Divide your data into 3 parts. The First is 70% of the data for training, the next 15% is for evaluation and the rest 15% is for testing the model (Percentage may vary accordingly). After training the model, check how it is performing on the eval set. If the performance is low, make changes and repeat the process. Then move to the testing part and get the predictions. There are many machine learning libraries out there. But in this blog, let's see only the basic working of NumPy, Pandas and Linear Algebra. I will cover more libraries in my future blogs.

## Exploring Numpy and Pandas:-

Just Knowing the functions and syntax of a library is not enough. We should know where and how to use a particular library. Let's take a car dataset and try to work with it. You can find the dataset [here](https://github.com/nivasgopi30/Machine-Learning-Engineering/blob/master/cars.csv).

First import the required libraries.
```
import numpy as np
import pandas as pd
``` 

In Pandas library, there are Series and DataFrame data structures. First, read the CSV file and store the data as a DataFrame(here cars_data is the name of our dataframe).

```
cars_data = pd.read_csv('cars.csv')
``` 

You can check the number of rows and columns in the dataset with the 'shape' attribute.

```
print(cars_data.shape)
print(len(cars_data))
``` 
**Note:- For output, please check out my repo. I will share the link below.**
Once observe the first few rows(by default it shows 5) of the data.

```
print(cars_data.head())
``` 

Let's say we want to find out the most three frequent cars in our dataset. We can use the collections module here. You can check the code for that:-

```
from collections import Counter

counter = Counter(cars_data['Make'])
print(counter.most_common(3))
``` 
Maybe you just want to know the number of unique models that the Audi brand has. There is a method called nunique() which will do that for us.

```
print(cars_data[cars_data['Make'] == 'Audi']['Model'].nunique())
``` 
You can also know the statistical measures of one or more columns.

```
print(cars_data['Engine Cylinders'].median())
print(cars_data['Engine Cylinders'].mode())
``` 
You can fill the null values in a column with the fillna() method. Here we are filling with 4.

```
print(cars_data['Engine Cylinders'].fillna(4.0))
``` 
That's it for this article.

If you are interested to see for more code and output, check out my GitHub repo [here].(https://github.com/nivasgopi30/Machine-Learning-Engineering/blob/master/Working%20with%20Numpy%20and%20Pandas.ipynb)

You can also join the ongoing ML Zoomcamp 2022 if you want to learn more. Huge shoutout to my favourite and amazing [Alexey Grigorev](https://twitter.com/Al_Grigor) for doing this amazing community work. #mlzoomcamp #learninginpublic

**Thank You.** Any comments and constructive criticism are welcome. I won't stop giving content like this. If you want more content, follow me on [Twitter](https://twitter.com/nivasgopi30) and [LinkedIn](https://www.linkedin.com/in/nivas-gopi-marella-4a6785208/).

