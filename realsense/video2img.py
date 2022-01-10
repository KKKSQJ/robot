import cv2
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', default='./video.avi')
    parser.add_argument('--out-path', default='./out-img')
    args = parser.parse_args()

    assert os.path.exists(args.video_path)
    jpeg_root = os.path.join(args.out_path, 'JPEGImage')
    os.makedirs(jpeg_root, exist_ok=True)

    vc = cv2.VideoCapture(args.video_path)
    c = 0

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        if frame is None:
            continue
        cv2.imwrite(os.path.join(jpeg_root, str(c)+'.jpg'), frame)
        print('Write img in {}'.format(os.path.join(jpeg_root, str(c)+'.jpg')))
        c += 1
        cv2.waitKey(1)
    vc.release()
