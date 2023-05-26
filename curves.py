import numpy as np
import matplotlib.pyplot as plt



TESTTASK = 'DMS'

lt = np.loadtxt

bls = [lt('blosses_onerun.txt')]
#bls = [lt('bl1.txt'), lt('bl2.txt'), lt('bl3.txt'), lt('bl4.txt'), lt('bl5.txt'), lt('bl6.txt')]


LEN = np.min([x.size for x in bls])
bl = np.vstack( [x[:LEN] for x in bls] )
print(LEN)


bl = .5 + .5 * bl

#bl = .5 + .5 * np.loadtxt('blosses_onerun.txt')

if(len(bl.shape)<2):   # If there is only a single run, add a singleton dimension
    bl = bl[None, :]
print(bl.shape)
ss = bl.shape[1]  # Number of generations

plt.figure(figsize=(4,4))

xr = np.arange(len(bl[0,:]))
plt.fill_between(xr[xr%10 != 0], np.quantile(bl, .25, axis=0).T[xr % 10 != 0], np.quantile(bl, .75, axis=0).T[xr % 10 != 0], color='b', alpha=.3)
plt.plot(xr[xr % 10 != 0], np.quantile(bl, .5, axis=0).T[xr % 10 != 0], 'b', label='Training tasks');
plt.fill_between(xr[::10], np.quantile(bl, .25, axis=0).T[0::10], np.quantile(bl, .75, axis=0).T[0::10], color='r', alpha=.3)
plt.plot(xr[::10], np.quantile(bl, .5, axis=0).T[0::10], 'r', label='Test task')


plt.xlabel('Generations')
plt.ylabel('% correct over last 100 trials')
plt.legend(loc='lower right')

plt.title('Test task: '+str(TESTTASK).upper())

plt.tight_layout()
plt.show()
