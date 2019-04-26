import matplotlib.pyplot as plt

def plot_ROC(tpr,fpr,auc,save=False):
    """Plots the receiver operating characteristic
    
    Args:
        tpr (np array): True positive rate
        fpr (np array): False positive rate
        auc (float): Area under the curve
        save (bool): Indicates if the plot should be stored
    """
    plt.figure(figsize=(6,6),dpi=100)
    plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], lw=2,color = [0,0,0], linestyle='--')
    plt.xlim([-0.005, 1.0])
    plt.ylim([0.0, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig("AUC Plot.png")