import sklearn
import codecs
import xlrd
import csv
import xlrd
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import math
import time

def test_faith():
    """The 1st part of project3"""
    data_1 = xlrd.open_workbook(r'G:\pyproj\EE511Proj3\oldfaithful.xlsx')    #Change the directory to file directory
    table = data_1.sheet_by_name(u'Sheet1')
    x = table.col_values(0)
    y = table.col_values(1)
    c = []
    for i in range(0,len(x)):
        c.append(table.row_values(i))
    print(c)
    [centroid,label,inertia] = cluster.k_means(c,2)
    print(centroid)
    print(label)
    print(inertia)
    for a in range(0,len(label)):
        plt.scatter(x[a],y[a],c = 'b')
    plt.xlabel('eruptions')
    plt.ylabel('waiting')
    plt.title('Raw Data')
    plt.show()
    for j in range(0,len(label)):
        if label[j] == 1:
            plt.scatter(x[j],y[j],marker = '.',c = 'r')
        elif label[j] == 0:
            plt.scatter(x[j],y[j],marker = '*',c = 'b')
    plt.xlabel('eruptions')
    plt.ylabel('waiting')
    plt.title('Clustering of Data')
    plt.show()
    plt.clf()

def em():
    """The 2nd part of project3"""
    np.random.seed()
    x = mixture.GaussianMixture(n_components = 2, covariance_type = 'full')
    s = []
    p = []
    q = []
    mu_1 = [1,1]
    mu_2 = [3,3]
    sigma_1 = ([1,0],[0,2])
    sigma_2 = ([1,0],[0,2])
    for i in range(0,150):
        c = np.random.multivariate_normal(mu_1,sigma_1)
        d = np.random.multivariate_normal(mu_2,sigma_2)
        s.append(c)
        p.append(d)
        q.append(c)
        q.append(d)
    for i in range(0,len(s)):
        plt.scatter(s[i][0],s[i][1],c = 'r')
        plt.scatter(p[i][0],p[i][1],c = 'b')
    plt.plot()
    plt.xlabel("The 1st dimension of GMM")
    plt.ylabel("The 2nd dimension of GMM")
    plt.title("2DGMM with 2 subpopulation in each dimension")
    plt.show()
    y = mixture.GaussianMixture(n_components = 2)
    start = time.clock()
    x.fit(q)
    end = time.clock()
    print("The time of EM is:")
    print(end - start)
    print("The weight of each Gaussian distribution is:")
    print(x.weights_)
    print("The covariance matrix of the distribution is:")
    print(x.covariances_)
    print("The mean point of 2-dimension Gaussian distribution is:")
    print(x.means_)
    data_1 = xlrd.open_workbook(r'G:\pyproj\EE511Proj3\oldfaithful.xlsx')
    table = data_1.sheet_by_name(u'Sheet1')
    m = table.col_values(0)
    c = []
    for i in range(0,len(m)):
        c.append(table.row_values(i))
    print(y.fit(c))
    print("The weight of each distribution in old faith data is:")
    print(y.weights_)
    print("The covariance matrix of the data is:")
    print(y.covariances_)
    print("The mean points of distribution are:")
    print(y.means_)

def clstr_txt(k):
    """The 3rd part of project3"""
    with open(r'G:\pyproj\EE511Proj3\nips-87-92.csv') as csvfile:    #Change the directory to file directory
        reader = csv.reader(csvfile)
        y = []
        for j,rows in enumerate(reader):
            for i in range(0,701):
                if j == i:
                    a = rows
                    y.append(a)
        del y[0]
        [centroid,label,inertia] = cluster.k_means(y,k)
        print(centroid)
        print(label)
        print(inertia)
        return math.log10(inertia)
    csvfile.close()

def clstr_txt3():
    with open(r'G:\pyproj\EE511Proj3\nips-87-92.csv') as csvfile:    #Change the directory to file directory
        reader = csv.reader(csvfile)
        y = []
        x = []
        b = []
        c = []
        for j,rows in enumerate(reader):
            for i in range(0,701):
                if j == i:
                    a = rows
                    y.append(a)
        del y[0]
        [centroid,label,inertia] = cluster.k_means(y,3)
        print(centroid)
        print(label)
        print(inertia)
        for m in range(0,len(label)):
            if label[m] == 0:
                x.append(y[m][0])
            elif label[m] == 1:
                b.append(y[m][0])
            elif label[m] == 2:
                c.append(y[m][0])
        print("The doc_id of the first cluster are:")
        print(x)
        print("The doc_id of the second cluster are:")
        print(b)
        print("The doc_id of the third cluster are:")
        print(c)
    csvfile.close()

test_faith()
em()
a = []
b = []
for k in range(2,10):
    c = clstr_txt(int(k))
    a.append(k)
    b.append(c)
plt.plot(a,b)
plt.xlabel('No. of clusters')
plt.ylabel('p-norm of cluster center to each point')
plt.title('Finding Elbow')
plt.show()
clstr_txt3()     #clustering in k = 3


