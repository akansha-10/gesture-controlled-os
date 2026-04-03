import cv2, time
idx = 0   # change to the working index if test_index found a different one
cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("VideoCapture.open failed for index", idx)
    raise SystemExit(1)
start = time.time()
print("Showing frames for up to 3 minutes. Press 'q' in the window to quit.")
while True:
    ok, frame = cap.read()
    print("read ok:", ok, "elapsed:", round(time.time()-start,1))
    if not ok:
        print("Read failed — empty frame")
        break
    cv2.imshow("testcam", frame)
    k = cv2.waitKey(1000) & 0xFF
    if k == ord('q') or time.time()-start > 180:
        print("exiting test")
        break
cap.release()
cv2.destroyAllWindows()
