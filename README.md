# Gradient Descent & Eve
Compare gradient descent algorithm to Eve.

## References

### Articles
Adam algorithm is based on the article : [Adam : A Method For Stochastic Optimization](https://arxiv.org/pdf/1412.6980v8.pdf) by Diederik P. Kingma and Jimmy Lei Ba.  
Eve algorithm is based on the article : [Improving Stochastic Gradient Descent With Feedback](https://arxiv.org/pdf/1611.01505v2.pdf) by Jayanth Koushik & Hiroaki Hayashi.

## Execution
To download data (at the first use), execute :  
```
python3.5 dataExtraction.py
```

To compute a comparison of the different algorithms, execute :  
```
python3.5 performances.py
```

All the needed algorithm are present in the gradientDescent.py file.

## Results
We compute the comparison on the dataset : [Ionosphere Data Set](http://archive.ics.uci.edu/ml/datasets/Ionosphere). The goal is to predict the class of the radar.  

Error after 1000 iterations :  
![Results](https://raw.githubusercontent.com/Jeanselme/Eve/master/Images/Results.png)

This Figure shows the imporvement implied by Adam algorithm which has an impressive error decrease. Moreover, it highlights the adavantage of an auto adaptative learning rate on training error.  

Moreover, the data are shuffled at the beginning of each iteration, that has a huge benefit on the total error.

## Libraries
Needs numpy, os, pandas and urllib.request. Executed with python3.5 on Linux
