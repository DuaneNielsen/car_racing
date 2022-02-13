import numpy as np
from statistics import mean
from matplotlib import pyplot as plt

start = 70
ep = 0
rms = np.load(f'../data/ep{ep}_error.npy')[start:]
ep_len = rms.shape[0]

lookahead = 128
lookbehind = 128
window_size = lookahead + lookbehind
rms_stack = []
max_error_stack, min_error_stack, mean_error_stack, current_pt = [], [], [], []

fig = plt.figure()
ax_err = fig.add_subplot(2, 1, 1)
ax_img = fig.add_subplot(2, 1, 2)

for t in range(ep_len):
    t_tail, t_middle, t_head = t - window_size, t - lookahead, t
    if np.isinf(rms[t]):
        max_error_stack.append(np.inf)
        min_error_stack.append(np.inf)
        mean_error_stack.append(np.inf)
        current_pt.append(np.inf)
    else:
        rms_stack.append(rms[t])
        current_pt.append(rms[t])
        max_error_stack.append(max(rms_stack))
        min_error_stack.append(min(rms_stack))
        mean_error_stack.append(mean(rms_stack))

    if t_tail >= 0:
        rms_stack.pop(0)
        ax_img.clear()
        seg = np.load(f'../data/road_sdfs/0/{ep}_{t_tail}.npz')['arr_0']
        ax_img.imshow(seg)



    if t > window_size:
        x = list(range(len(max_error_stack)))
        ax_err.clear()
        ax_err.plot(x, max_error_stack)
        ax_err.plot(x, min_error_stack)
        ax_err.plot(x, mean_error_stack)
        ax_err.plot(x, current_pt)
        plt.pause(0.02)
plt.show()