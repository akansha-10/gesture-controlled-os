import cv2, time
for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok, frame = cap.read()
    print(f"index {i}: ok={ok}")
    if ok and frame is not None:
        try:
            h,w = frame.shape[:2]
            print(f"  frame shape: {w}x{h}")
        except:
            pass
    cap.release()
    time.sleep(0.2)
