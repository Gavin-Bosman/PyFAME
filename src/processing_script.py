import pyfame as pf
import matplotlib.pyplot as plt
import numpy as np

in_dir = "C:\\Users\\gavin\\Desktop\\PyFAME\\data\\Video_Song_Actors_01-24\\Video_Song_Actor_01\\Actor_01\\01-02-02-01-01-02-01.mp4"
out_dir = "C:\\Users\\gavin\\Desktop\\PyFAME\\images\\"

onset = 500
offset = 4000
rise = 1500
fall = 1500

times = np.arange(0, 6000, 10)
weights = [pf.timing_gaussian(x, onset, offset, rise, fall) for x in times]

plt.figure(figsize=(10, 6))
plt.plot(times, weights, label='Gaussian Timing', color='darkorange', linewidth=2)
plt.title('Gaussian Timing Function', fontsize=16)
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('Timing Weight (0 to 1)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()