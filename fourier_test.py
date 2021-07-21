import numpy as np
import cv2
import logging
from matplotlib import pyplot as plt


"""
modern cuda programming
CUDA C

"""


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise Exception(f"cap.read() for frame returned {ret}")
    return frame


def read_all_frames(cap):
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"cap.read() for frame returned {ret}")
            break
        frames.append(frame)
    return frame


def canny_one_frame(fr1):
    t1 = 100
    t2 = 200
    aptsize = 3
    t1p, t2p, asp = t1,t2,aptsize
    while True:
        ret = cv2.Canny(fr1, t1, t2, apertureSize=aptsize)
        cv2.imshow('ack', ret)
        if (t1,t2,aptsize) != (t1p,t2p,asp):
            print(f'{t1=}, {t2=}, {aptsize=}')
        t1p, t2p, asp = t1,t2,aptsize
        k = cv2.waitKey(30) & 0xff
        if k == 27: # escape
            break
        elif k == 113: # q
            if t1 > 0:
                t1 -= 1
        elif k == 119: # w
            t1 += 1
        elif k == 97:  # a
            if t2 > 0:
                t2 -= 1
        elif k == 115: # s
            t2 += 1
        elif k == 122: # z
            if aptsize > 3:
                aptsize -= 2
        elif k == 120: # x
            if aptsize < 7:
                aptsize += 2


def fourier_one_frame_numpy(fr1):
    # fr1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
    fr1 = cv2.Canny(fr1, 100, 200)
    f = np.fft.fft2(fr1)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(2,3,1),plt.imshow(fr1, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,3,2),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    rows, cols = fr1.shape
    crow,ccol = rows//2, cols//2
    fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    plt.subplot(2,3,4),plt.imshow(fr1, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,3,5),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,3,6),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

    plt.show()


def fourier_one_frame_cv(fr1):
    fr1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
    # fr1 = cv2.Canny(fr1, 100, 200)
    dft = cv2.dft(np.float32(fr1), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    plt.subplot(2,3,1),plt.imshow(fr1, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,3,2),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    rows, cols = fr1.shape
    crow,ccol = rows//2, cols//2
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    plt.subplot(2,3,4),plt.imshow(fr1, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,3,5),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,3,6),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

    plt.show()


def fourier_video_numpy_interactive(cap):
    text = []
    for t in range(10):
        for c in (0,1):
            for s in range(100):
                text.append((f"t{t:02}c{c}s{s:03}",read_frame(cap)))
    idx = 0
    length = len(text)
    while True:
        fr1 = cv2.cvtColor(text[idx][1], cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(fr1), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        # print(magnitude_spectrum.shape, magnitude_spectrum.dtype)
        dist1 = cv2.normalize(magnitude_spectrum, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.putText(dist1, text[idx][0],(2,25), 3, 1, (255,255,255))
        cv2.imshow('beanslol',dist1)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            idx -= 1
            if idx < 0:
                idx = 0
        elif k == ord('d'):
            idx += 1
            if idx >= length:
                idx = length-1

# write output to histogram for distribution of frequencies
def fourier_hist_numpy(cap):
    magspec = []
    while True:
        try:
            fr1 = read_frame(cap)
        except:
            break
        fr1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(fr1), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        dist1 = cv2.normalize(magnitude_spectrum, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        magspec.append(dist1)
    # magspec = np.array(magspec)
    lorbo = tuple(chunks(magspec, 100))
    color=(
        'blue',
        'green',
        'red',
        'orange',
        'gray',
        'black',
        'cyan',
        'magenta',
        'indigo',
        'violet'
    )
    print(len(lorbo))
    for i,col in enumerate(color):
        hist = cv2.calcHist(lorbo[i],[0],None,[256],[0,256])
        plt.plot(hist,color=col)
        plt.xlim([0,256])
    plt.show()


def fourier_video_numpy(cap):
    for t in range(5):
        for c in (0,1):
            for s in range(100):
                fr1 = read_frame(cap)
                fr1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
                dft = cv2.dft(np.float32(fr1), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)
                magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
                # print(magnitude_spectrum.shape, magnitude_spectrum.dtype)
                dist1 = cv2.normalize(magnitude_spectrum, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                cv2.putText(dist1, f"t{t:02}c{c}s{s:03}",(2,25), 3, 1, (255,255,255))
                cv2.imshow('beanslol',dist1)
                k = cv2.waitKey(30) & 0xFF
                if k == 27:
                    return


def main():
    sbj = 'callen_exp03'
    cap = cv2.VideoCapture(f'{sbj}/{sbj}.mkv')
    fourier_hist_numpy(cap)
    # fourier_video_numpy_interactive(cap)
    # canny_one_frame(read_frame(cap))
    # fourier_one_frame_numpy(read_frame(cap))
    cap.release()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
