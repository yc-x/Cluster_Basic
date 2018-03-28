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
    mu_2 = [4,4]
    sigma_1 = ([6,0],[0,6])
    sigma_2 = ([6,0],[0,6])
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
    a = np.linspace(-7,14)
    b = np.linspace(-7,14)
    A,B = np.meshgrid(a,b)
    Q = np.array([A.ravel(), B.ravel()]).T
    z = - x.score_samples(Q)
    z = z.reshape(A.shape) 
    plt.contour(a,b,z, levels=np.logspace(0, 1, 10))
    plt.xlabel("The 1st dimension of GMM")
    plt.ylabel("The 2nd dimension of GMM")
    plt.title("2DGMM with 2 subpopulation in each dimension")
    plt.show()
    y = mixture.GaussianMixture(n_components = 2)
    data_1 = xlrd.open_workbook(r'G:\pyproj\EE511Proj3\oldfaithful.xlsx')
    table = data_1.sheet_by_name(u'Sheet1')
    m = table.col_values(0)
    n = table.col_values(1)
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
    
    
def em_of():
	"""EM on old_faith data"""
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
    for j in range(0,len(label)):
       if label[j] == 1:
           plt.scatter(x[j],y[j],marker = '.',c = 'r')
       elif label[j] == 0:
           plt.scatter(x[j],y[j],marker = '*',c = 'b')
    plt.xlabel('eruptions')
    plt.ylabel('waiting')
    plt.title('The comparison between kmeans clustering and EM model')
    k = mixture.GaussianMixture(n_components = 2)
    print(k.fit(c))
    a = np.linspace(0,6)
    b = np.linspace(40,100)
    A,B = np.meshgrid(a,b)
    Q = np.array([A.ravel(), B.ravel()]).T
    z = - k.score_samples(Q)
    z = z.reshape(A.shape) 
    plt.contour(a,b,z, levels = np.logspace(0, 1, 10))
    plt.show()
    
def clstr_txt(k):
    """The 3rd part of project3"""
    with open(r'G:\pyproj\EE511Proj3\nips-87-92.csv') as csvfile:    #Change the directory to file directory
        reader = csv.reader(csvfile)
        y = []
        for j,rows in enumerate(reader):
            for i in range(0,701):
                if j == i:
                    a = rows
                    del a[0]
                    y.append(a)
        del y[0]
        [centroid,label,inertia] = cluster.k_means(y,k)
        print(centroid)
        print(label)
        print(inertia)
        return math.log10(inertia)
    csvfile.close()

def clstr_txt8():
    with open(r'G:\pyproj\EE511Proj3\nips-87-92.csv') as csvfile:    #Change the directory to file directory
        reader = csv.reader(csvfile)
        y = []
        for j,rows in enumerate(reader):
            for i in range(0,701):
                if j == i:
                    a = rows
                    del a[0]
                    y.append(a)
        del y[0]
        [centroid,label,inertia] = cluster.k_means(y,8)
        print(centroid)
        print(label)
        print(inertia)
        return label
    csvfile.close()

def out_clstr8():
    a = clstr_txt8()
    with open(r'G:\pyproj\EE511Proj3\nips-87-92.csv') as csvfile:    #Change the directory to file directory
        reader = csv.reader(csvfile)
        c_1 = []
        c_2 = []
        c_3 = []
        c_4 = []
        c_5 = []
        c_6 = []
        c_7 = []
        c_8 = []
        b = [row[0] for row in reader]
        del b[0]
        for m in range(0,len(a)):
            if a[m] == 0:
                c_1.append(b[m])
            elif a[m] == 1:
                c_2.append(b[m])
            elif a[m] == 2:
                c_3.append(b[m])
            elif a[m] == 3:
                c_4.append(b[m])
            elif a[m] == 4:
                c_5.append(b[m])
            elif a[m] == 5:
                c_6.append(b[m])
            elif a[m] == 6:
                c_7.append(b[m])
            elif a[m] == 7:
                c_8.append(b[m])
        print("The doc_id of the 1st cluster are:")
        print(c_1)
        print("The doc_id of the 2nd cluster are:")
        print(c_2)
        print("The doc_id of the 3rd cluster are:")
        print(c_3)
        print("The doc_id of the 4th cluster are:")
        print(c_4)
        print("The doc_id of the 5th cluster are:")
        print(c_5)
        print("The doc_id of the 6th cluster are:")
        print(c_6)
        print("The doc_id of the 7th cluster are:")
        print(c_7)
        print("The doc_id of the 8th cluster are:")
        print(c_8)
    csvfile.close()

test_faith()
em()
a = []
b = []
x = input("Please input the upbound of k you want to test:")
for k in range(2,int(x) + 1):
    c = clstr_txt(int(k))
    a.append(k)
    b.append(c)
plt.plot(a,b)
plt.xlabel('No. of clusters')
plt.ylabel('log of p-norm of cluster center to each point')
plt.title('Finding Elbow')
plt.show()
out_clstr8()   #clustering in k = 8



