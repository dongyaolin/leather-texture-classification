import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


y = pd.read_csv('./fig/loss_log.csv')
plt.figure(figsize=(8, 9), dpi=200)
x = np.array(range(len(y))).astype(int)
plt.plot(x, y.iloc[:, -1])
plt.xlabel('iter')
plt.ylabel('loss')
# plt.xticks(x, [str(i*100) for i in x/100])

plt.savefig(f'./loss_log_{time.strftime("%d%H%m")}.jpg')
plt.pause(5)
