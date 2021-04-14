import numpy as np
import cv2

frames = []
# with np.load('callen_exp02_img.npz') as n:
#     frames = np.concatenate((
#         n['focus'][0],n['relax'][0],
#         n['focus'][1],n['relax'][1],
#         n['focus'][2],n['relax'][2],
#         n['focus'][3],n['relax'][3],
#         n['focus'][4],n['relax'][4]))
for i in range(100):
    frames.append(cv2.imread(f'callen_exp03/ft00s{i:03}.webp'))
for i in range(100):
    frames.append(cv2.imread(f'callen_exp03/rt00s{i:03}.webp'))

# elinux.org cameras for jetsons
# dense
frame1 = frames.pop(0)
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1]=255
for frame2 in frames:
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

# sparse
# feature_params = dict(
#     maxCorners = 100,
#     qualityLevel = 0.3,
#     minDistance = 7,
#     blockSize = 7
# )

# lk_params = dict(
#     winSize = (15,15),
#     maxLevel = 2,
#     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
# )

# color = np.random.randint(0,255,(100,3))
# old_frame = frames.pop(0)
# old_gray = cv2.Canny(old_frame,60,240,True)#cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# mask = np.zeros_like(old_frame)

# for frame in frames:
#     frame_gray = cv2.Canny(frame,50,240,True)#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     p1,st,err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

#     good_new = p1[st==1]
#     good_old = p0[st==1]

#     for i,(new,old) in enumerate(zip(good_new,good_old)):
#         a,b = new.ravel()
#         c,d = old.ravel()
#         mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(),2)
#         frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
#     img = cv2.add(frame,mask)

#     cv2.imshow('frame',img)
#     k = cv2.waitKey(30000) & 0xff
#     if k == 27:
#         break
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
