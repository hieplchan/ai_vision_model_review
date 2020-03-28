import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import csv

LOG_DIR = './'

process_time_tmp = 0.0
process_time = []
process_time_histogram = []
num = []
N_point = 0
n_bins = 100

benchmark_name = 'mobilenetv2'

print('start')
################# CALCULATE RESULT #################
with open('time.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        num.append(int(N_point))
        process_time.append(float(row[0]))
        process_time_histogram.append(float(row[0]))
        N_point = N_point + 1

process_time_max = max(process_time)
process_time_min = min(process_time)
process_time_avr = np.mean(process_time)
process_time_std = np.std(process_time)

################# TIME PLOT #################
plt.plot(num,process_time,label='process_time - Thread')
plt.xlabel('sample - Max: ' + str(process_time_max) + ' - Avr: ' + str(process_time_avr) + ' - Standard Deviation: ' +str(process_time_std))
plt.ylabel('process_time [ms]')
plt.title('Process Time Plot')
plt.legend()
plt.savefig(LOG_DIR + 'time_' + benchmark_name + '.png', transparent=False, bbox_inches='tight', pad_inches=0)
plt.show()

plt.clf()

################# PLOT HISTOGRAM #################
N, bins, patches = plt.hist(process_time_histogram, bins=n_bins)
fracs = N.astype(float) / N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
plt.xlabel('process_time [ms] - Max: ' + str(process_time_max) + ' - Avr: ' + str(process_time_avr) + ' - Standard Deviation: ' +str(process_time_std))
plt.ylabel('number of sample')
plt.title('Process Time Histogram - Thread')
plt.legend()
plt.savefig(LOG_DIR + 'hist_' + benchmark_name + '.png', transparent=False, bbox_inches='tight', pad_inches=0)
plt.show()

print(N_point)
