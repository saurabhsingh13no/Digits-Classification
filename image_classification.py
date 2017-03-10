import matplotlib.pylab as plt
import numpy as np
from sklearn import datasets
digits=datasets.load_digits()
x=digits.images[0]
# plt.imshow(x,cmap=plt.cm.gray_r,interpolation="nearest")
# plt.show()
def distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

random_digit_no=1705
x_test=digits.images[random_digit_no]
training_lenth=len(digits.images)-200
dist=np.zeros(training_lenth)

"""Code written for testing"""
for i in range(training_lenth):
    dist[i]=distance(digits.images[i],x_test)

predict=np.argmin(dist)
print ("x_test was {} and algorithm predicted it to be {}".format(digits.target[random_digit_no],digits.target[predict]))


"""Determing accuracy of the model"""

test_sample_size=len(digits.images)-training_lenth
error=0
for i in range(training_lenth,len(digits.images)):  #Running the algorithm over all test samples
    for j in range(training_lenth):
        dist[j]=distance(digits.images[j],digits.images[i])
    predicted_image_index=np.argmin(dist)
    if digits.target[predicted_image_index]!=digits.target[i]:
        error+=1
accuracy=100.0*(test_sample_size-error)/test_sample_size
print ("The accuracy of the model over {} test sample is {}%".format(test_sample_size,accuracy))
