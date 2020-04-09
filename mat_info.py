#!/usr/bin/python3.6
import numpy
import scipy.io
import sys

fn = sys.argv[1]
mat = scipy.io.loadmat(fn)
for x in mat:
    k = mat[x]
    print(f'{x}:')
    if x[0] != '_':
        print(f'   type = {type(k)}, shape = {k.shape}, dtype = {k.dtype}')
