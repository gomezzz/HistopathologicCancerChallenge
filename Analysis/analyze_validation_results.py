import sys
sys.path.append('../') 

from sklearn import metrics
import numpy as np
from Utils.plotting_utils import *

def analyze_validation_results(X,labels,predictions):
    """Analyzes validation results. Compute confusion matrix, AUC/ROC, Errors vs. Brightness, Failure cases

    Args:
        X (np array): Corresponding images (NxHxWxC)
        labels (np array): Ground-truth labels for the images (N)
        predictions (np array): Predicted labels for the images (N)
    """
    N = X.shape[0]

    ####################################################
    # Print confusion matrix
    ####################################################
    tn, fp, fn, tp = metrics.confusion_matrix(labels,np.round(predictions)).ravel()
    print("True positives = {} ({}%)".format(tp,tp * 100 / N))
    print("False positives = {} ({}%)".format(fp,fp * 100 / N))
    print("True negatives = {} ({}%)".format(tn,tn * 100 / N))
    print("False negatives = {} ({}%)".format(fn,fn * 100 / N))

    ####################################################
    # Plot ROC, Compute AUC
    ####################################################
    fpr,tpr,_ = metrics.roc_curve(labels,predictions)
    auc = metrics.auc(fpr,tpr)
    plot_ROC(tpr,fpr,auc)
    
    # Get indices of tps, fps, ...
    rounded_pred = np.round(predictions) 
    tps = np.multiply(labels,rounded_pred) 
    tns = np.multiply(1-labels,1-rounded_pred)
    fps = np.maximum(0,rounded_pred - labels)
    fns = np.maximum(0,labels - rounded_pred)

    ####################################################
    # Plot TP,FP,TN,FN vs Image brightness
    ####################################################
    tp_k,fp_k,tn_k,fn_k = [],[],[],[]
    image_count_k = []
    image_brightness = np.mean(X,axis=(1,2,3))
    xtics = range(100,255,1)
    for threshold in xtics:
        #count images with more than threshold 255 pixels
        image_brighter_than_k = (image_brightness > threshold)
        tp_k.append(np.logical_and(tps,image_brighter_than_k).sum() / tp) 
        fp_k.append(np.logical_and(fps,image_brighter_than_k).sum() / fp) 
        tn_k.append(np.logical_and(tns,image_brighter_than_k).sum() / tn) 
        fn_k.append(np.logical_and(fns,image_brighter_than_k).sum() / fn) 
        image_count_k.append(image_brighter_than_k.sum() / N)
        
    fig = plt.figure(figsize=(6,3),dpi=200)
    plt.plot(xtics,image_count_k)
    plt.plot(xtics,tp_k)
    plt.plot(xtics,fp_k)
    plt.plot(xtics,tn_k)
    plt.plot(xtics,fn_k)
    plt.legend(("Images brighter than threshold","Cumulative TPs","Cumulative FPs","Cumulative TNs","Cumulative FNs"),loc=1,prop={'size': 6})
    plt.title("Cumulative (TP,FP,TN,FN) distributions vs mean brightness")
    plt.xlabel("Mean image brightness threshold")
    plt.ylabel("Cumulative frequency")
    plt.tight_layout()
    plt.show()

    ####################################################
    # Plot TP,FP,TN,FN vs Nr. of bright pixels
    ####################################################
    tp_k,fp_k,tn_k,fn_k = [],[],[],[]
    image_count_k = []
    bright_pixel_count = (X == 255).sum(axis=(1,2,3))
    xtics = range(0,5000,100)
    for threshold in xtics:
        #count images with more than threshold 255 pixels
        image_brighter_than_k = (bright_pixel_count > threshold)
        tp_k.append(np.logical_and(tps,image_brighter_than_k).sum() / tp) 
        fp_k.append(np.logical_and(fps,image_brighter_than_k).sum() / fp) 
        tn_k.append(np.logical_and(tns,image_brighter_than_k).sum() / tn) 
        fn_k.append(np.logical_and(fns,image_brighter_than_k).sum() / fn) 
        image_count_k.append(image_brighter_than_k.sum() / N)
        
    fig = plt.figure(figsize=(6,3),dpi=200)
    plt.plot(xtics,image_count_k)
    plt.plot(xtics,tp_k)
    plt.plot(xtics,fp_k)
    plt.plot(xtics,tn_k)
    plt.plot(xtics,fn_k)
    plt.legend(("Images over threshold","Cumulative TPs","Cumulative FPs","Cumulative TNs","Cumulative FNs"),loc=1,prop={'size': 6})
    plt.title("Cumulative (TP,FP,TN,FN) distributions vs bright pixels")
    plt.xlabel("Nr. of bright pixels threshold")
    plt.ylabel("Cumulative frequency")
    plt.tight_layout()
    plt.show()

    ####################################################
    # Investigate model confidence (How close to 0/1 for TP,FP,TN,FN)
    ####################################################
    nr_of_bins = 25
    fig,axs = plt.subplots(2,2,sharex=False,sharey=False,figsize=(6,6),dpi=150)

    axs[0,0].hist(predictions[np.where(tns)],bins=nr_of_bins,density=True)
    axs[0,1].hist(predictions[np.where(tps)],bins=nr_of_bins,density=True)
    axs[1,0].hist(predictions[np.where(fns)],bins=nr_of_bins,density=True)
    axs[1,1].hist(predictions[np.where(fps)],bins=nr_of_bins,density=True)
    
    #Set image labels
    axs[0,0].set_title("True Negatives");
    axs[0,1].set_title("True Positives");
    axs[1,0].set_title("False Negatives");
    axs[1,1].set_title("False Postives");
    axs[0,0].set_ylabel("Frequency")        
    axs[1,0].set_ylabel("Frequency")
    axs[1,0].set_xlabel("Prediction")
    axs[1,1].set_xlabel("Prediction")
    fig.tight_layout()
    plt.show()