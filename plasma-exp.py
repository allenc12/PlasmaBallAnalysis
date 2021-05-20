import time
from math import floor
import os
import pathlib
import cv2
import numpy
# import matplotlib.pyplot as plt
# from scipy.signal import detrend
# from scipy.stats import ranksums


class Trial:

    def __init__(self, cam, subject: str = None,
                 num_trials: int = 5,
                 num_samples: int = 100):
        if subject is not None:
            self.subject = subject
        else:
            subject = input("Enter subject name: ")
            if len(subject) == 0:
                subject = 'exp'
            for i in range(10000):
                if not pathlib.Path(f'{subject}{i:05d}').exists():
                    self.subject = f'{subject}{i:05d}'
                    break
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
            else:
                self.concentrate = False
                if not self.quiet:
                    print("Relax")
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
        )
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        out = cv2.VideoWriter(f'{self.subject}/{self.subject}.mkv', fourcc, 20.0, (640,480))
        for t in range(self.nTrials):
            for c in (0,1):
                for s in range(self.nSamples):
                    if c:
                        out.write(self.framesFocus[t][s])
                    else:
                        out.write(self.framesRelax[t][s])
        out.release()


def main():
    if os.name == 'nt':
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(0)
    # warm up the camera
    for _ in range(5):
        cam.read()
    """
    nT=20;nS=100;
    moA=[random.uniform(121.3040,123.8315) for _ in range(nT*nS)]
    cA=[[0]*nS if i%2==0 else [1]*nS for i in range(nT)]
    cA=[item for sublist in cA for item in sublist]
    tA=[random.uniform(6.6020,6.6890) for _ in range(nT)]
    """
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    tr = Trial(cam)

    # TODO: get subject name in a nicer way

    # ani = animation.FuncAnimation(fig, tr.nop)
    # plt.draw()
    tr.run_trial()
    tr.save_output()
    #plt.show()

    cam.release()

    input("Press any key to exit...")


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
