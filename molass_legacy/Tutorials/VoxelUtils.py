# coding: utf-8
"""
    VoxelUtils.py

    from https://www.mahirokazuko.com/entry/2019/06/24/232831
"""
import colorsys
import collections
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def int2hue(integer, in_max=255, out_max=255, is_alpha=False):
    if is_alpha:
        return tuple( float(v * out_max) for v in  colorsys.hsv_to_rgb(integer/in_max, 1, 1) ) + (out_max,)
    else:
        return tuple( float(v * out_max) for v in  colorsys.hsv_to_rgb(integer/in_max, 1, 1) )

def flatten(xs):
    result = []
    for x in xs:
        if isinstance(x, collections.Iterable) and not isinstance(x, dict)and not isinstance(x, str):
            result.extend(flatten(x))
        else:
            result.append(x)
    return result

def draw_voxel(ax, tensor):
    tensor = tensor.transpose(2,0,1) # 軸を入れ替える。※ 機械学習で画像を扱うときによく使う軸の順番にしただけで、それ以外の深い意味は無い
    # fig = plt.figure(figsize=(5, 5), dpi=100) # 図の大きさを変えられる
    # ax = fig.gca(projection='3d')
    org_shape = tensor.shape

    tensor_min = tensor.min() # すべての要素が正になるように、もし負の値があったらその絶対値を全要素に加算する
    if tensor_min < 0:
        tensor = tensor - tensor_min

    eps = 1e-10
    tensor = tensor + eps # 要素の値が0だと表示できないことがあるので、それを回避するために微小な数を与える

    tensor_max = tensor.max()
    colors = [int2hue(x, in_max=tensor_max, out_max=1.0, is_alpha=True) for x in flatten(tensor)]
    colors = np.reshape(colors, org_shape + (4,)) # 各座標に、RGBAを表すサイズ4の配列を埋め込むためシェイプに4を追加する

    if False:
        # 立方体になるように軸を調整
        xsize, ysize, zsize = tensor.shape
        max_range = np.array([xsize, ysize, zsize]).max() * 0.5
        mid_x = xsize * 0.5
        mid_y = ysize * 0.5
        mid_z = zsize * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 目盛りを削除
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_zticklabels(), visible=False)

    ax.voxels(tensor, facecolors=colors, linewidth=0.3, edgecolor='k')
    #  plt.show()
