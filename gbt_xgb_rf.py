import numpy as np
import gc
import os
import sys
import time
import random
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

#===== Designation of target year / month =====#
monthA = ["200611", "200612"]                  #
monthB = ["200701", "200702"]                  #
monthC = ["200703", "200704"]                  #
monthD = ["200705", "200706"]                  #
monthE = ["200707", "200708"]                  #
monthF = ["200709", "200710"]                  #
monthG = ["200711", "200712"]                  #
monthH = ["200801", "200802"]                  #
monthI = ["200803", "200804"]                  #
monthJ = ["200805", "200806"]                  #
monthK = ["200807", "200808"]                  #
monthL = ["200809", "200810"]                  #
monthM = ["200811", "200812"]                  #
monthN = ["201311", "201312"]                  #
monthO = ["201401", "201402"]                  #
monthP = ["201403", "201404"]                  #
monthQ = ["201405", "201406"]                  #
monthR = ["201407", "201408"]                  #
monthS = ["201409", "201410"]                  #
monthT = ["201411", "201412"]                  #
monthU = ["201501", "201502"]                  #
monthV = ["201503", "201504"]                  #
monthW = ["201505", "201506"]                  #
monthX = ["201507", "201508"]                  #
monthY = ["201509", "201510"]                  #
monthZ = ["201511", "201512"]                  #
#==============================================#



#================================= CREATE DATA =================================#
def create_data(month):                                                         #
    pos = []                                                                    #
    neg = []                                                                    #
    for mon in month:                                                           #
        for i in range(1, 32):                                                  #
            data = mon + ('%02d.txt' %i)                                        #
            if os.path.exists(data):                                            #
                for line in open(data, 'rb'):                                   #
                    ls = line.split()                                           #
                    a = int(ls[17])     #Label                                  #
                    b = float(ls[0])    #Session time                           #
                    c = float(ls[2])    #Number of Send Byte                    #
                    d = float(ls[3])    #Number of                              #
                    e = float(ls[4])    #                                       #
                    f = float(ls[5])    #                                       #
                    g = float(ls[6])    #                                       #
                    h = float(ls[7])    #                                       #
                    i = float(ls[8])    #                                       #
                    j = float(ls[9])    #                                       #
                    k = float(ls[10])   #                                       #
                    l = float(ls[11])   #                                       #
                    m = float(ls[12])   #                                       #
                    #n = float(ls[19])   #Sender                                #
                    #o = float(ls[21])   #Destination                           #
                    if a > 0:                                                   #
                        a = 1                                                   #
                        pos.append([b, c, d, e, f, g, h, i, j, k, l, m])        #
                    else:                                                       #
                        a = 0                                                   #
                        neg.append([b, c, d, e, f, g, h, i, j, k, l, m])        #
    random.shuffle(pos)                                                         #
    random.shuffle(neg)                                                         #
    return pos, neg                                                             #
#===============================================================================#

process_start = time.time()
start = time.time()
pos, neg = create_data(monthT)
end = time.time()
print("===== Training Data Reading Time =====")
print("{0}[s]".format(end - start))
start = time.time()
post, negt = create_data(monthU)
end = time.time()
print("===== Test Data Reading Time =====")
print("{0}[s]".format(end - start))

gacc = 0
gpre = 0
gtpr = 0
gfpr = 0
gf1 = 0
gtime = 0

xacc = 0
xpre = 0
xtpr = 0
xfpr = 0
xf1 = 0
xtime = 0

racc = 0
rpre = 0
rtpr = 0
rfpr = 0
rf1 = 0
rtime = 0

gb = GradientBoostingClassifier(loss = "exponential",
                                learning_rate = 0.3,
                                n_estimators = 400,         
                                random_state = None)


xgb = xgb.XGBClassifier(learning_rate =0.45,
                        n_estimators=90,
                        max_depth=6,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.75,
                        colsample_bytree=0.8,
                        objective= 'binary:logistic',
                        nthread=4,                   
                        scale_pos_weight=1,
                        seed=27)

rf = RandomForestClassifier(n_estimators = 100)

for i in range(10):
    print("\n------------------------- TRIAL {0} -------------------------".format(i + 1))
    
    random.shuffle(pos)
    random.shuffle(neg)
    X = neg[0:10000] + pos[0:10000]
    X_train = np.array(X)
    y = [0]*10000 + [1]*10000
    y_train = np.array(y)

    random.shuffle(post)
    random.shuffle(negt)
    Xt = negt[0:10000] + post[0:10000]
    X_test = np.array(Xt)
    yt = [0]*10000 + [1]*10000
    y_test = np.array(yt)


#========================= scikit-learn GBT ========================#
    gb = gb.fit(X_train, y_train)                                   #
    y_train_pred = gb.predict(X_train)                              #
    test_start = time.time()                                        #
    y_test_pred = gb.predict(X_test)                                #
    test_end = time.time()                                          #
    gb_train = accuracy_score(y_train, y_train_pred)                #
    tp, fn, fp, tn = confusion_matrix(y_test, y_test_pred).ravel()  #
    TP = float(tp)                                                  #
    FP = float(fp)                                                  #
    TN = float(tn)                                                  #
    FN = float(fn)                                                  #
    acc = 100*(TP+TN)/(TP+FP+TN+FN)                                 #
    pre = 100*TP/(TP+FP)                                            #
    tpr = 100*TP/(TP+FN)                                            #
    fpr = 100*FP/(FP+TN)                                            #
    F1 = 2*pre*tpr/(pre+tpr)                                        #
    timer = test_end - test_start                                   #
    gacc += acc                                                     #
    gpre += pre                                                     #
    gtpr += tpr                                                     #
    gfpr += fpr                                                     #
    gf1 += F1                                                       #
    gtime += timer                                                  #
    print("\n===== scikit-learn Gradient Boosting =====")           #
    print("TP\tFP\tTN\tFN")                                         #
    print("{0}\t{1}\t{2}\t{3}".format(tp, fp, tn, fn))              #
    print("\nsklearn GBT train = {0}".format(gb_train))             #
    print("sklearn GBT Accuracy = {0}".format(acc))                 #
    print("sklearn GBT Precision = {0}".format(pre))                #
    print("sklearn GBT Recall (TPR) = {0}".format(tpr))             #
    print("sklearn GBT FPR = {0}".format(fpr))                      #
    print("sklearn GBT F1-Score = {0}".format(F1))                  #
    print("Test time = {0}".format(timer))                          #
#===================================================================#


#============================== XGBoost ============================#
    xgb = xgb.fit(X_train, y_train)                                 #
    y_train_pred = xgb.predict(X_train)                             #
    xtest_start = time.time()                                       #
    y_test_pred = xgb.predict(X_test)                               #
    xtest_end = time.time()                                         #
    xgb_train = accuracy_score(y_train, y_train_pred)               #
    tp, fn, fp, tn = confusion_matrix(y_test, y_test_pred).ravel()  #
    TP = float(tp)                                                  #
    FP = float(fp)                                                  #
    TN = float(tn)                                                  #
    FN = float(fn)                                                  #
    acc = 100*(TP+TN)/(TP+FP+TN+FN)                                 #
    pre = 100*TP/(TP+FP)                                            #
    tpr = 100*TP/(TP+FN)                                            #
    fpr = 100*FP/(FP+TN)                                            #
    F1 = 2*pre*tpr/(pre+tpr)                                        #
    timer = xtest_end - xtest_start                                 #
    xacc += acc                                                     #
    xpre += pre                                                     #
    xtpr += tpr                                                     #
    xfpr += fpr                                                     #
    xf1 += F1                                                       #
    xtime += timer                                                  #
    print("\n===== XGBoost =====")                                  #
    print("TP\tFP\tTN\tFN")                                         #
    print("{0}\t{1}\t{2}\t{3}".format(tp, fp, tn, fn))              #
    print("\nXGBoost train = {0}".format(xgb_train))                #
    print("XGBoost Accuracy = {0}".format(acc))                     #
    print("XGBoost Precision = {0}".format(pre))                    #
    print("XGBoost Recall (TPR) = {0}".format(tpr))                 #
    print("XGBoost FPR = {0}".format(fpr))                          #
    print("XGBoost F1-Score = {0}".format(F1))                      #
    print("Test time = {0}".format(timer))                          #
#===================================================================#


#=========================== Random Forest =========================#
    rf = rf.fit(X_train, y_train)                                   #
    y_train_pred = rf.predict(X_train)                              #
    rtest_start = time.time()                                       #
    y_test_pred = rf.predict(X_test)                                #
    rtest_end = time.time()                                         #
    rf_train = accuracy_score(y_train, y_train_pred)                #
    tp, fn, fp, tn = confusion_matrix(y_test, y_test_pred).ravel()  #
    TP = float(tp)                                                  #
    FP = float(fp)                                                  #
    TN = float(tn)                                                  #
    FN = float(fn)                                                  #
    acc = 100*(TP+TN)/(TP+FP+TN+FN)                                 #
    pre = 100*TP/(TP+FP)                                            #
    tpr = 100*TP/(TP+FN)                                            #
    fpr = 100*FP/(FP+TN)                                            #
    F1 = 2*pre*tpr/(pre+tpr)                                        #
    timer = rtest_end - rtest_start                                 #
    racc += acc                                                     #
    rpre += pre                                                     #
    rtpr += tpr                                                     #
    rfpr += fpr                                                     #
    rf1 += F1                                                       #
    rtime += timer                                                  #
    print("\n===== Random Forest =====")                            #
    print("TP\tFP\tTN\tFN")                                         #
    print("{0}\t{1}\t{2}\t{3}".format(tp, fp, tn, fn))              #
    print("\nRandom Forest train = {0}".format(rf_train))           #
    print("Random Forest Accuracy = {0}".format(acc))               #
    print("Random Forest Precision = {0}".format(pre))              #
    print("Random Forest Recall (TPR) = {0}".format(tpr))           #
    print("Random Forest FPR = {0}".format(fpr))                    #
    print("Random Forest F1-Score = {0}".format(F1))                #
    print("Test time = {0}".format(timer))                          #
#===================================================================#


print("\n\n= = = = = = = = = = RESULT = = = = = = = = = =")

print("\n==== scikit-learn GBT ====")
print("\tAccuracy = {0}".format(gacc/10))
print("\tPrecision = {0}".format(gpre/10))
print("\tTrue Positive Rate = {0}".format(gtpr/10))
print("\tFalse Positive Rate = {0}".format(gfpr/10))
print("\tF1-Score = {0}".format(gf1/10))
print("\tTest time = {0}".format(gtime/10))

print("\n==== XGBoost ====")
print("\tAccuracy = {0}".format(xacc/10))
print("\tPrecision = {0}".format(xpre/10))
print("\tTrue Positive Rate = {0}".format(xtpr/10))
print("\tFalse Positive Rate = {0}".format(xfpr/10))
print("\tF1-Score = {0}".format(xf1/10))
print("\tTest time = {0}".format(xtime/10))

print("\n==== Random Forest ====")
print("\tAccuracy = {0}".format(racc/10))
print("\tPrecision = {0}".format(rpre/10))
print("\tTrue Positive Rate = {0}".format(rtpr/10))
print("\tFalse Positive Rate = {0}".format(rfpr/10))
print("\tF1-Score = {0}".format(rf1/10))
print("\tTest time = {0}".format(rtime/10))

print("\n= = = = = = = = = = = = = = = = = = = = = = =")

process_end = time.time()
print("\n===== PROCESS TIME =====")
time = process_end - process_start
minute = int(time / 60)
second = time % 60
print("{0}:{1}".format(minute, second))
