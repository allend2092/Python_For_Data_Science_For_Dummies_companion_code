# Learning to Use Python Fast

## Loading data
from sklearn.datasets import load_boston
boston = load_boston()
X, y = boston.data, boston.target

## Training a model
from sklearn.linear_model import LinearRegression
hypothesis = LinearRegression(normalize=True)
hypothesis.fit(X, y)

## Viewing a result
print(hypothesis.coef_)
