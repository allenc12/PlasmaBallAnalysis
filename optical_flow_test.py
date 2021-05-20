import numpy as np
import cv2
import logging
from pathlib import Path

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        logging.warning(f"cap.read() for frame returned {ret}")
    return frame


# https://github.com/allenc12/PlasmaBallAnalysis
# https://www.nvidia.com/en-us/gtc/keynote/

# elinux.org cameras for jetsons
# dense
def preview_cam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame1 = cap.read()
    if not ret:
        logging.warning(f"frame1 read returned {ret}")
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1]=255
    while True:
        ret, frame2 = cap.read()
        if not ret:
            logging.warning(f"frame2 read returned {ret}")
        nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow: np.ndarray = cv2.calcOpticalFlowFarneback(prvs,nxt,None,0.5,3,15,3,5,1.2,0)
        mag,ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2',rgb)
        k=cv2.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = nxt
    cap.release()


def get_flow_video(pth: Path = Path('./callen_exp03')):
    dat = pth / 'flow.npz'
    if dat.exists():
        d = np.load(dat)['flows']
        return d
    vid = pth / f'{pth.parts[0]}.mkv'
    cap = cv2.VideoCapture(str(vid))
    frame = read_frame(cap)
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev,nxt,None,0.5,3,15,3,5,1.2,0)
        frames.append(flow)
        prev = nxt
    cap.release()
    frames = np.array(frames)
    np.savez_compressed(dat, flows=frames)
    return frames


def small_flow(filename: str = './callen_exp02/callen_exp02.mkv'):
    cap = cv2.VideoCapture(filename)
    frame1 = read_frame(cap)
    frame2 = read_frame(cap)
    print(f'frame1: {frame1.shape}, {frame1.dtype}')
    # frame1 = cv2.imread('./callen_exp03/ft00s000.png')
    # frame2 = cv2.imread('./callen_exp03/ft00s001.png')
    prv = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    print(f'prv: {prv.shape}, {prv.dtype}')
    flow = cv2.calcOpticalFlowFarneback(prv,nxt,None,0.5,3,15,3,5,1.2,0)
    print(f'flow: {flow.shape}, {flow.dtype}')
    cap.release()
    return flow


def convert_viddy():
    video_filename = './callen_exp01/callen_exp01.mkv'
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640,480))
    frames = []
    for t in range(5):
        for ch in 'rf':
            for s in range(100):
                fr = cv2.imread(f'./callen_exp01/{ch}t{t:02d}s{s:03d}.png')
                frames.append(fr)
                out.write(fr)
    out.release()
    cap = cv2.VideoCapture(video_filename)
    ii = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        equ = np.array_equal(frame, frames[ii])
        if not equ:
            print(f"frame != frames[{ii}] : {equ}")
        ii += 1
    cap.release()


def cvt_flow2hsv(flow):
    shape = flow.shape[:-1]+(3,)
    hsv = np.zeros(shape, dtype=np.uint8)
    mag,ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return rgb


def flow_mean():
    pth = Path('./callen_exp03')
    frames = get_flow_video(pth)[0:99]
    u = np.average(frames[...,0], axis=0)
    v = np.average(frames[...,1], axis=0)
    shape = frames[0].shape[:-1]+(3,)
    hsv = np.zeros(shape, dtype=np.uint8)
    mag,ang = cv2.cartToPolar(u, v)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',rgb)
    while True:
        k=cv2.waitKey(30) & 0xff
        if k == 27:
            break


def main():
    x=range(100)
    print(next(x))
    for i in x:
        print(f"x:{i}")
    # small_flow()
    # flow_mean()
    pass


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
