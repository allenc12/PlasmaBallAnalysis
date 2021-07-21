import io
import tarfile
from pathlib import Path
import time
import os
import logging

import cv2
import numpy as np
# from matplotlib import pyplot as plt


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
            self.focus = [list() for _ in range(self.trials)]
            self.relax = [list() for _ in range(self.trials)]
            memb = m.pop(tuple(map(lambda x: x.name.endswith(f'{self.subject}.mkv'), m)).index(True))
            cap = cv2.VideoCapture(tf.extractfile(memb))
            for t in range(self.trials):
                for c in (0,1):
                    for s in range(self.samples):
                        ret, frame = cap.read()
                        if not ret:
                            logging.warning(f"cap.read() returned {ret} for t{t} c{c} s{s}")
                        if c:
                            self.focus[t].append(frame)
                        else:
                            self.relax[t].append(frame)
            logging.info(f"done reading tarfile: {self.focus.size}")

    def read_data(self):
        with np.load(f"{self.path}/dat.npz") as dat:
            self.trials = dat['trials']
            self.samples = dat['samples']
        self.focus = [list() for _ in range(self.trials)]
        self.relax = [list() for _ in range(self.trials)]
        cap = cv2.VideoCapture(f'{self.subject}/{self.subject}.mkv')
        for t in range(self.trials):
            for c in (0,1):
                for s in range(self.samples):
                    ret, frame = cap.read()
                    if not ret:
                        logging.warning(f"cap.read() returned {ret} for t{t} c{c} s{s}")
                    if c:
                        self.focus[t].append(frame)
                    else:
                        self.relax[t].append(frame)
        self.focus = np.array(self.focus)
        self.relax = np.array(self.relax)
        logging.info(f"done reading data: focus<shape={self.focus.shape}, dtype={self.focus.dtype}>")

    def calc_flows(self):
        pth = Path(f"{self.path}/flow.npy")
        if pth.exists():
            return np.load(pth)
        frgen = self.seq_frames()
        prev = cv2.cvtColor(next(frgen), cv2.COLOR_BGR2GRAY)
        for i in range((self.trials*self.samples*2)-1):
            nxt = cv2.cvtColor(next(frgen), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev,nxt,None,0.5,3,15,3,5,1.2,0)
            self.flows.append(flow)
            prev = nxt
        self.flows = np.array(self.flows)
        logging.info(f"finshed calculating optical flow: {self.flows.shape}, {self.flows.dtype}")

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

    def split_flows(self):
        u = np.empty(self.flows.shape[:-1]+(4,),dtype=np.uint8)
        v = np.empty_like(u.shape, dtype=u.dtype)
        u = self.flows[...,0]
        v = self.flows[...,1]
        return u,v


# XXX: UPDATE IMAGE PATH FSTRINGS IN analysis.py AND plasma-exp.py
def main():
    pd = PlasmaData('callen_exp03','callen_exp03')
    pd.read_data()
    pd.calc_flows()
    # write to file
    # fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    # out = cv2.VideoWriter('callen_exp03/flows_.mkv', fourcc, 20.0, (640,480))
    # for i in range(len(pd.flows)):
    #     out.write(pd.flows[i,:,:,0])
    # out.release()


def oldmain():
    pd = PlasmaData('callen_exp03','callen_exp03')
    pd.read_data()
    pd.calc_flows()
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
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = nxt


if __name__ == "__main__":
    main()
