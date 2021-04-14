import cv2, numpy as np, os

img = cv2.imread('callen_exp03/callen_exp03_focus-t00s000.png')

np.savez_compressed('tst.npz', img=img)
with np.load('tst.npz') as f:
    cmp = f['img']
cmp_stat = os.stat('tst.npz')
print(f"Compressed numpy array    : {np.array_equiv(img, cmp):1}, Size:{cmp_stat.st_size:9d}")

_, buf = cv2.imencode('.tiff', img)
tif = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
print(f"Tiff Image file           : {np.array_equiv(img, tif):1}, Size:{buf.size:9d}")

_, buf = cv2.imencode('.png', img)
png = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
print(f"PNG Image file            : {np.array_equiv(img, png):1}, Size:{buf.size:9d}")

_, buf = cv2.imencode('.webp', img)
web = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
print(f"WebP Image file           : {np.array_equiv(img, web):1}, Size:{buf.size:9d}")

_, buf = cv2.imencode('.bmp', img)
bmp = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
print(f"Windows Bitmap file       : {np.array_equiv(img, bmp):1}, Size:{buf.size:9d}")

_, buf = cv2.imencode('.jpg', img)
jpg = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
print(f"JPEG file                 : {np.array_equiv(img, jpg):1}, Size:{buf.size:9d}")
