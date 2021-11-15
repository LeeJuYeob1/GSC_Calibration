import numpy as np
import cv2
objp = np.zeros((6 * 7, 3), np.float32)
objp[ : , : 2] = np.mgrid[0 : 7, 0 : 6].T.reshape(-1, 2)
#3D 포인트
objpoints = []
#2D 포인트
imgpoints = []
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cam = cv2.VideoCapture(0)
(w, h) = (int(cam.get(4)), int(cam.get(3)))
while(True):
    _ , frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #패턴찾기
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    if ret == True:
        objpoints.append(objp)
        #정확도 올리기
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        #패턴 그리기
        cv2.drawChessboardCorners(frame, (7, 6), corners, ret)
        cv2.imshow('Find Chessboard', frame)
        #패턴을 찾으면 아무키를 눌러주세요.
        cv2.waitKey(0)
    cv2.imshow('Find Chessboard', frame)
    print ("Number of chess boards find:", len(imgpoints))
    if cv2.waitKey(1) == 27:
        break
#켈리브레이션: 카메라 행렬, 왜곡 계수, 회전, 이동벡터 반환
ret, oldMtx, coef, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                      gray.shape[: : -1], None, None)
#왜곡 개선
newMtx, roi = cv2.getOptimalNewCameraMatrix(oldMtx, coef, (w, h), 1, (w, h))
#카메라 파라미터값 저장
print ("Original Camera Matrix:\n", oldMtx)
print ("Optimal Camera Matrix:\n", newMtx)
np.save("Original camera matrix", oldMtx)
np.save("Distortion coefficients", coef)
np.save("Optimal camera matrix", newMtx)
cam.release()
cv2.destroyAllWindows()

