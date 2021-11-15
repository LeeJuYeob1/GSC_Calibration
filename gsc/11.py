import cv2
import numpy as np
import glob

# 체커보드 사이즈 지정
CHECKERBOARD = (7, 7)
#K 평균 군집화 알고리즘 조건 생성
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
#켈리브레이션 조건 생성
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
#체커보드를 찾을 OBJ 포인트 준비
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#이미지 파일 읽어오기
images = glob.glob('3/1 *.jpg')
for fname in images:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    #흑백으로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 체커보드 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # Image points (after refinin them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (7, 6), corners, ret)
        cv2.imshow('Find Chessboard', img)
N_OK = len(objpoints)
#왜곡 계수가 들어갈 배열 생성
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
#카메라 켈리브레이션
rms, _, _, _, _ = \
cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

#어안렌즈의 경우 왜곡이 심해 켈리브레이션 진행 후 사각형이 아닌 부분을 제거 하기 위한  balance 값 설정
DIM=_img_shape[::-1]
balance=0
dim2=None
dim3=None

#수정할 파일
img = cv2.imread("3/1 (1).jpg")
#DIM 값을 통한 이미지 확대
dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
if not dim2:
    dim2 = dim1
if not dim3:
    dim3 = dim1
scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
#왜곡 보정
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

#데이터 값 저장
data = {'dim1': dim1,
        'dim2':dim2,
        'dim3': dim3,
        'K': np.asarray(K).tolist(),
        'D':np.asarray(D).tolist(),
        'new_K':np.asarray(new_K).tolist(),
        'scaled_K':np.asarray(scaled_K).tolist(),
        'balance':balance}

import json
with open("fisheye_calibration_data.json", "w") as f:
    json.dump(data, f)

#웹캠 출력
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
(w, h) = (int(cam.get(4)), int(cam.get(3)))
print(w, h)
while(True):
    _, frame = cam.read()
    if not _:
        break
    #왜곡 없애기
    map3, map4 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img1 = cv2.remap(frame, map3, map4, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #왜곡 개선 영상
    #cv2.imshow("",undistorted_img1)

    #왜곡 개선 전, 후 비교
    cv2.imshow("Original vs Undistortion", np.hstack([frame, undistorted_img1]))
    key = cv2.waitKey(1)
    if key == 27:
        break


cam.release()
#수정된 이미지출력
#cv2.imshow("undistorted", undistorted_img)
#undistorted_img1 = cv2.remap(img2, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#cv2.imshow("none undistorted", undistorted_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()