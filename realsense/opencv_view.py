## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
import json
import os
import argparse
import time

import png
import pyrealsense2 as rs
import numpy as np
import cv2


def make_directories(folder):
    if not os.path.exists(folder + "/JPEGImages/"):
        os.makedirs(folder + "/JPEGImages/")
    if not os.path.exists(folder + "/depth/"):
        os.makedirs(folder + "/depth/")


def main(arg):
    s = False
    if arg.save and arg.folder:
        s = True
        make_directories(arg.folder)

    FileName = 0
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start pipeline
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Color Intrinsics
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }
    print('camera_parameters:', camera_parameters)

    if s:
        with open(arg.folder + '/intrinsics.json', 'w') as fp:
            json.dump(camera_parameters, fp)

    align_to = rs.stream.color
    align = rs.align(align_to)

    T_start = time.time()
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        d = np.asanyarray(aligned_depth_frame.get_data())
        c = np.asanyarray(color_frame.get_data())

        # Visualize count down

        if time.time() - T_start > 5:
            filecad = arg.folder + "/JPEGImages/%s.jpg" % FileName
            filedepth = arg.folder + "/depth/%s.png" % FileName
            if s:
                cv2.imwrite(filecad, c)
            with open(filedepth, 'wb') as f:
                writer = png.Writer(width=d.shape[1], height=d.shape[0],
                                    bitdepth=16, greyscale=True)
                zgray2list = d.tolist()
                if s:
                    writer.write(f, zgray2list)

            FileName += 1

        if time.time() - T_start > arg.record_length + 5:
            pipeline.stop()
            break

        if time.time() - T_start < 5:
            cv2.putText(c, str(5 - int(time.time() - T_start)), (240, 320), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4,
                        (0, 0, 255), 2, cv2.LINE_AA)
        if time.time() - T_start > arg.record_length:
            cv2.putText(c, str(arg.record_length + 5 - int(time.time() - T_start)), (240, 320),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('COLOR IMAGE', c)

        # press q to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pipeline.stop()
            break

    # Release everything if job is finished
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', type=bool, default=True, help='whether store img')
    parser.add_argument('-f', '--folder', type=str, default='rs', help='path to store img')
    parser.add_argument('--record-length', type=int, default=np.inf, help='Recording duration')
    arg = parser.parse_args()

    main(arg)
