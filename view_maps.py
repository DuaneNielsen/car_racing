from matplotlib import pyplot as plt
import numpy as np
import pathlib


if __name__ == '__main__':
    files = pathlib.Path('data').glob('*_map.npy')
    maps = [np.load(f.absolute()) for f in files]
    fig = plt.figure(figsize=(18, 18))
    axes = fig.subplots(1)

    for i, amap in enumerate(maps):
        axes.clear()
        axes.imshow(amap[500:1500, 750:1750])
        fig.suptitle(f'map {i}')
        plt.pause(7.0)
    plt.show()

