##################################################################
# Solver for linear equation tranformation for points set registration,
# after the corresponds stage
# Can be used for RANSAC after and counting the number of successes
##################################################################

import numpy as np
import math

# generate examples for test
def gen_targets(S = [[1,1,1],[2,1,1], [1,2,1]], ang=10,dist=5):
    """
    Computes the transformation and returns it
    :param S:
    :return: mat-transformation, sp-soure, tar-target, where tar = mat*sp
    """
    mat = np.zeros((2,3), dtype=np.float32)
    ang = math.pi*ang/180
    cs = math.cos(ang)
    sn = math.sin(ang)
    mat[0,0] = mat[1,1] = cs
    mat[1,0] = -sn
    mat[0,1] = sn
    mat[:,2] = dist
    sp = np.array(S)
    tar = np.matmul(mat,sp)
    return mat,sp, tar

# Solve the correspondence problem for registration getting source and target
def solver(src, tar):
    XX = np.matmul(src, src.T)
    tx = np.matmul(tar,src.T)
    XXI = np.linalg.pinv(XX)
    mat = np.matmul(tx, XXI)
    return mat


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ang=5
    del_ang = 6
    dst=5
    del_dst = 7
    EPS = 1e-5
    for k in range(30):
        mat,src,tar = gen_targets(ang=ang, dist=dst)
        mat_r = solver(src,tar)
        print(f'mat:{mat}\nmat_r:{mat_r}')
        diff = np.abs(mat_r - mat)
        ids = diff > EPS
        if np.sum(ids) > 0:
            print(f'\n*********** difference not zero: \n{diff}')

        print('\n---------------------------------------------\n')
        ang += del_ang
        dst += del_dst

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
