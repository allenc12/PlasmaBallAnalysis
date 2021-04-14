import io
import tarfile
import time
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


class PlasmaData:

    def __init__(self, path = '', subject='', read_init = False):
        self.focus = []
        self.relax = []
        self.basic_analysis = None
        self.subject: str = subject
        self.path: str = path
        if read_init:
            self.read_data()

    def read_data(self):
        if self.path.endswith('.tar'):
            tar_bytes = open(self.path, 'rb').read()
            iob = io.BytesIO(tar_bytes)
            with tarfile.open(fileobj=iob) as tf:
                m = tf.getmembers()
                memb = m.pop(tuple(map(lambda x: x.name.endswith('z'), m)).index(True))
                self.basic_analysis = np.load(tf.extractfile(memb)) # maybe cleanup
                mbarr = np.array_split(m, self.basic_analysis['trials']*2)
                with np.nditer(mbarr, op_flags=['readwrite']) as it:
                    for x in it:
                        x[...] = cv2.imdecode(tf.extractfile(x).read())
                mbarr = np.array(mbarr)
                self.focus = mbarr[0::2]
                self.relax = mbarr[1::2]
        else:
            self.basic_analysis = np.load(f"{self.path}/dat.npz")
            self.focus = [list() for i in range(self.basic_analysis['trials'])]
            self.relax = [list() for i in range(self.basic_analysis['trials'])]
            for t in range(5):
                for s in range(100):
                    self.focus[t].append(cv2.imread(f"{self.path}/ft{t:02d}s{s:03d}.webp"))
                    self.relax[t].append(cv2.imread(f"{self.path}/rt{t:02d}s{s:03d}.webp"))
        print(f"done reading data: {len(self.focus)=}")

    def seq_frames(self):
        for trial in range(self.basic_analysis['trials']):
            for i in range(2):
                if i:
                    for j in range(self.basic_analysis['samples']):
                        yield self.relax[trial][j]
                else:
                    for j in range(self.basic_analysis['samples']):
                        yield self.focus[trial][j]
# XXX: UPDATE IMAGE PATH FSTRINGS IN analysis.py AND plasma-exp.py
# unfold plasma ball

pd = PlasmaData('callen_exp03','callen_exp03')
pd.read_data()
frame1 = next(pd.seq_frames())
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1]=255
for frame2 in pd.seq_frames():
    nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,nxt,None,0.5,3,15,3,5,1.2,0)
    mag,ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',rgb)
    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break
    prvs = nxt

# img = pd.focus[0][0]
# img = cv2.imread('./callen_exp02/ft00s000.webp',0)
# edges = cv2.Canny(img, 60, 240, True)

# orb = cv2.ORB_create()
# kp = orb.detect(img,None)
# kp,des = orb.compute(img,kp)
# img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()

# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap='gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()
