import time
from math import floor
import os
import cv2
import numpy
# import matplotlib.pyplot as plt
# from scipy.signal import detrend
# from scipy.stats import ranksums


class Trial:

    def __init__(self, cam, subject: str = input("Enter subject name: "),
                 num_trials: int = 10,
                 num_samples: int = 30):
        self.subject = subject
        self.nTrials = num_trials
        self.nSamples = num_samples
        self.cam = cam
        self.concentrate = True
        self.time_delta = []
        self.framesFocus = []
        self.framesRelax = []
        self.quiet = False

    def get_sample(self, i: int = 0):
        _, o = self.cam.read()
        time.sleep(0.1)
        # _, r = cam.read()
        if self.concentrate:
            if i == 0:
                self.framesFocus.append([])
            self.framesFocus[-1].append(o)
        else:
            if i == 0:
                self.framesRelax.append([])
            self.framesRelax[-1].append(o)

    def run_trial(self):
        for _ in range(self.nTrials):
            if not self.concentrate:
                self.concentrate = True
                if not self.quiet:
                    print("Concentrate")
                    # time.sleep(1)
            else:
                self.concentrate = False
                if not self.quiet:
                    print("Relax")
                    # time.sleep(1)
            t1 = time.monotonic_ns()
            for i in range(self.nSamples):
                self.get_sample(i)
            t2 = time.monotonic_ns()
            elapsed = floor(t2 - t1)
            self.time_delta.append(elapsed)
            # print(f"elapsed: {self.elapsed}")

    def save_output(self):
        """Save compressed trial data
        TODO: don't clobber output, perhaps auto-rename files
              save data into a .tar file, instead of loose files
        """
        os.mkdir(self.subject)
        numpy.savez_compressed(
            f'{self.subject}/data.npz',
            subject=numpy.array(self.subject),
            trials=numpy.full((), self.nTrials, dtype=numpy.int32),
            samples=numpy.full((), self.nSamples, dtype=numpy.int32),
            focus=self.framesFocus,
            relax=self.framesRelax,
        )
        # for i,tf in enumerate(self.framesFocus):
        #     for j,f in enumerate(tf):
        #         cv2.imwrite(f"{self.subject}/ft{i:02d}s{j:03d}.webp", f)
        # for i,tf in enumerate(self.framesRelax):
        #     for j,f in enumerate(tf):
        #         cv2.imwrite(f"{self.subject}/rt{i:02d}s{j:03d}.webp", f)


def main():
    cam = cv2.VideoCapture(0)
    # warm up the camera
    for _ in range(5):
        cam.read()
    # style.use('fivethirtyeight')
    # max(moA) 123.8315
    # min(moA) 121.3040
    """
    nT=20;nS=100;
    moA=[random.uniform(121.3040,123.8315) for _ in range(nT*nS)]
    cA=[[0]*nS if i%2==0 else [1]*nS for i in range(nT)]
    cA=[item for sublist in cA for item in sublist]
    tA=[random.uniform(6.6020,6.6890) for _ in range(nT)]
    """
    #fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    tr = Trial(cam)

    # TODO: get subject name in a nicer way

    # ani = animation.FuncAnimation(fig, tr.nop)
    # plt.draw()
    tr.run_trial()
    tr.save_output()
    #plt.show()

    cam.release()
    cv2.destroyAllWindows()

    input("Press any key to exit...")


if __name__ == "__main__":
    main()
