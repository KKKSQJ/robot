import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-root', type=str, default='rs/JPEGImages')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--size', default=(640, 480))
    parser.add_argument('--video-name', default='video.avi')

    args = parser.parse_args()

    assert os.path.exists(args.img_root)
    filelist = os.listdir(args.img_root)

    video = cv2.VideoWriter(args.video_name, cv2.VideoWriter_fourcc('I', '4', '2', '0'), args.fps, args.size)

    for item in tqdm(filelist):
        if item.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
            path = os.path.join(args.img_root, item)
            img = cv2.imread(path)
            img = cv2.resize(img, args.size)
            video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print('The video in this path: {}'.format(os.getcwd() + '/' + args.video_name))
