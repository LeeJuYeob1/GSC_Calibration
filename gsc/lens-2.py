import numpy as np
import cv2
oldMtx = np.load("Original camera matrix.npy")
coef = np.load("Distortion coefficients.npy")
newMtx = np.load("Optimal camera matrix.npy")
#웹 캠 띄우기
cam = cv2.VideoCapture(0)
(w, h) = (int(cam.get(4)), int(cam.get(3)))
while(True):
    _, frame = cam.read()
    #왜곡 없애기
    undis = cv2.undistort(frame, oldMtx, coef, newMtx)
    #왜곡 개선 전, 후 비교
    cv2.imshow("Original vs Undistortion", np.hstack([frame, undis]))
    key = cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
