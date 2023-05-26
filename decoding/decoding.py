import numpy as np; import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

print("This shows clipped graphs, BUT the image files will be OK!")

print('ATTENTION: we create graphs for decoding target, stimulus 1, and stimulus 2!')

#print(r.shape, s.shape, t.shape)
# (500, 70, 50, 2) (500, 2, 400) (500, 400)
# r has only the first and last trial

NBINPUTNEURONS = 7 
NBOUTPUTNEURONS =  6
NBTRIALS = 400


for numfig, figname in enumerate(['target', 'stim1',  'stim2']):
    fig = plt.figure(figsize=(5,5)); 

    ff,  axes = plt.subplots(2,2)
    ax = axes[0,0]

    print("Making figure", figname)

    for numgen in range(2):

        if numgen == 1:
            r = np.load('allresps.npy') ; s = np.load('allstims.npy')  ;  t = np.load('alltgts.npy')  
        else:
            r = np.load('allresps.npy.0'); s = np.load('allstims.npy.0'); t = np.load('alltgts.npy.0') 

        for numtrial in range(2):

            numplot =  1 + 2*numgen + numtrial
            print(numplot, "/", 2*2)
            plt.subplot(2,2, numplot)
            plt.gca().set_title('Gen '+str(numgen*1000)+' / Trial '+str(numtrial*NBTRIALS), fontsize=10)

            # Which trial are we looking at - first (well, actually 9th or 29th - last before onset of plasticity) or last?
            if numtrial == 0:
                rt = 0;    st = 29;  tt = 29 # Older ones use 29 as the "first" trial
                #rt = 0;    st = 9;  tt = 9
            else:
                rt = 1;    st = NBTRIALS - 1;  tt = NBTRIALS - 1

            allvals = []
            for timepoint_train in range(50):
                if timepoint_train % 10 == 9:
                    print(timepoint_train+1, '/ 50')
                vals_thistrainpoint = []
                
                #for timepoint_test in range(5):  # faster, for debugging
                for timepoint_test in range(50):


                    if numfig == 0:
                        # predicting target
                        y = (t[:, tt] -  .5) * 2
                    elif numfig  == 1:
                        # predicting first stimulus
                        y = (s[:, 0, st] - .5) * 2
                    elif numfig == 2:
                        # predicting second stimulus
                        y = (s[:, 1, st] - .5) * 2

                        
                    x_test = r[125:250, NBINPUTNEURONS:-NBOUTPUTNEURONS, timepoint_test, rt] 
                    y_test = y[125:250]
                    x_test = x_test - np.mean(x_test, axis=0)
                    x_test = x_test / (1e-8 + np.std(x_test, axis=0))

                    score = 0

                    nbtrainsets = 3  # ideally but not necessarily a divisor of 125, smaller=faster
                    for numtrain in range(nbtrainsets):

                        setsize =  125 // nbtrainsets

                        x_train = r[numtrain*setsize:(numtrain+1)*setsize, NBINPUTNEURONS:-NBOUTPUTNEURONS, timepoint_train, rt] 
                        y_train = y[numtrain*setsize:(numtrain+1)*setsize]

                        # Normalizing data to allow sklearn fitting 
                        x_train = x_train - np.mean(x_train, axis=0)
                        x_train = x_train / (1e-8 + np.std(x_train, axis=0))

                        traind1 = np.mean(x_train[y_train>0, :], axis=0)
                        traind2 = np.mean(x_train[y_train<0, :], axis=0)


                        cc1 = np.corrcoef(np.vstack((traind1, x_test)))[0, 1:]
                        cc2 = np.corrcoef(np.vstack((traind2, x_test)))[0, 1:]
                        choice = 2.0 * (cc1 > cc2) - 1.0
                        
                        score = score + np.mean(y_test == choice)
                    score = score / nbtrainsets


                    vals_thistrainpoint.append(score)
                allvals.append(vals_thistrainpoint)

            allvals = np.array(allvals)


            plt.imshow(allvals); plt.axhline(y=25,color='b', ls=":"); plt.axhline(y=35,color='b', ls=":"); plt.axvline(x=25,color='b', ls=":"); plt.axvline(x=35,color='b', ls=":")


            plt.xticks(np.arange(9,50,10), labels=[str(z) for z in  20*(1 + np.arange(9,50,10))] )
            plt.yticks(np.arange(9,50,10), labels=[str(z) for z in  20*(1 + np.arange(9,50,10))] )
            plt.clim(0, 1)
            if numgen == 1:
                plt.xlabel('Train time (ms)')
            if numtrial == 0:
                plt.ylabel('Test time (ms)')
            if numtrial == 1:
                plt.colorbar(); 

    #fig.suptitle('Changes due to lifetime learning')
    #fig.supylabel('Changes due to evolution')
    plt.tight_layout()
    if numfig == 0:
        ax.text(60, -20 ,'Changes due to lifetime learning', ha="center", va="center",size=10, color='b')
        ax.text(-38, 56 ,'Changes due to evolution', rotation=90,  ha="center", va="center",size=10, color='b')
        ax.annotate('', xytext=(-.2, 1.3), xycoords='axes fraction', xy=(3, 1.3),
                    arrowprops=dict(arrowstyle="->", color='b'))
        ax.annotate('', xytext=(-.6, 1), xycoords='axes fraction', xy=(-.6, -1.3),
                    arrowprops=dict(arrowstyle="->", color='b'))
    plt.show()
    plt.savefig("image_"+figname+".png",bbox_inches='tight',dpi=200)


