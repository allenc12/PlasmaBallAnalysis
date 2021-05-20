import io
import tarfile
from pathlib import Path
import time
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


class PlasmaData:

    def __init__(self, path = '', subject='', read_init = False):
        self.focus = None
        self.relax = None
        self.flows = []
        self.basic_analysis = None
        self.subject: str = subject
        self.path: str = path
        self.trials = 0
        self.samples = 0
        if read_init:
            self.read_data()

    def read_tar(self):
        tar_bytes = open(self.path, 'rb').read()
        iob = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=iob) as tf:
            m = tf.getmembers()
            memb = m.pop(tuple(map(lambda x: x.name.endswith('dat.npz'), m)).index(True))
            with np.load(tf.extractfile(memb)) as dat:
                self.trials = dat['trials']
                self.samples = dat['samples']
            self.focus = [list() for i in range(self.trials)]
            self.relax = [list() for i in range(self.trials)]
            memb = m.pop(tuple(map(lambda x: x.name.endswith(f'{self.subject}.mkv'), m)).index(True))
            cap = cv2.VideoCapture(tf.extractfile(memb))
            for t in range(self.trials):
                for c in (0,1):
                    for s in range(self.samples):
                        ret, frame = cap.read()
                        if c:
                            self.focus[t].append(frame)
                        else:
                            self.relax[t].append(frame)
            print(f"done reading tarfile: {self.focus.size}")

    def read_data(self):
        with np.load(f"{self.path}/dat.npz") as dat:
            self.trials = dat['trials']
            self.samples = dat['samples']
        self.focus = [list() for i in range(self.trials)]
        self.relax = [list() for i in range(self.trials)]
        cap = cv2.VideoCapture(f'{self.subject}/{self.subject}.mkv')
        for t in range(self.trials):
            for c in (0,1):
                for s in range(self.samples):
                    ret, frame = cap.read()
                    if c:
                        self.focus[t].append(frame)
                    else:
                        self.relax[t].append(frame)
        self.focus = np.array(self.focus)
        self.relax = np.array(self.relax)
        print(f"done reading data: focus<shape={self.focus.shape}, dtype={self.focus.dtype}>")

    def calc_flows(self):
        pth = Path(f"{self.path}/flow.npz")
        d = None
        if pth.exists():
            with np.load(pth) as dat:
                d = dat['flows']
            return d
        frgen = self.seq_frames()
        prev = cv2.cvtColor(next(frgen), cv2.COLOR_BGR2GRAY)
        for i in range((self.trials*self.samples*2)-1):
            nxt = cv2.cvtColor(next(frgen), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev,nxt,None,0.5,3,15,3,5,1.2,0)
            self.flows.append(flow)
            prev = nxt
        self.flows = np.array(self.flows)
        print(f"finshed calculating optical flow: {self.flows.shape}, {self.flows.dtype}")

    def calc_means(self):
        self.fmeans = []
        self.rmeans = []
        for t in range(self.trials):
            for c in (0,1):
                if c:
                    self.fmeans.append(np.average(self.focus[t]))
                else:
                    self.rmeans.append(np.average(self.relax[t]))

    def seq_frames(self):
        for trial in range(self.trials):
            for c in (0,1):
                for j in range(self.samples):
                    if c:
                        yield self.relax[trial][j]
                    else:
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
