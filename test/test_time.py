import pickle
import time
import numpy as np
import ipdb 
st = ipdb.set_trace
nppath = "/scratch/pointcloud.npz"
ppath = "/scratch/02958343_30456a19dc588d9461c6c61410fc904b.p"

# np.savez(open("/scratch/02958343_30456a19dc588d9461c6c61410fc904b.npz", "wb"), *a)
# st()git
sta = time.time()
for i in range(100):
    a = np.load(nppath)
    del a
print("numpy time: ", time.time()-sta)

sta = time.time()

for i in range(100):
    # a = pickle.load(open(ppath, "rb"))
    # a = np.load(ppath)
    # a=np.load('/scratch/02958343_30456a19dc588d9461c6c61410fc904b.npy', allow_pickle=True)
    a=np.load('/home/mprabhud/tmp/temp.npy', allow_pickle=True)
    # st()
    del a
print("pickle time: ", time.time()-sta)