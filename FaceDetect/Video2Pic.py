import cv2


VC = cv2.VideoCapture('test.mp4')
if VC.isOpened():
    rval, frame = VC.read()
else:
    rval = False
c = 1
while rval:
    rval, frame = VC.read()
    cv2.imwrite('./frames/%s.jpg' % c, frame)
    c += 1
    cv2.waitKey(1)
VC.release()