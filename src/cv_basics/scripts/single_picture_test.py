#!/usr/bin/env python3
import argparse
import os
import cv2
import numpy as np
import mmcv
import torch

from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.utils.general_utils import Timer
from mmdet.models.detectors.condlanenet import CondLanePostProcessor

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

SIZE = (800, 320)

import PIL.Image
import PIL.ImageDraw
from tools.condlanenet.common import COLORS, parse_lanes

#ros part
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'checkpoint', default=None, help='test config file path')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args( [
                "./robotaxi.py",
                "./model/culane_small.pth",
            ])
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def adjust_result(lanes, crop_bbox, img_shape, tgt_shape=(590, 1640)):

    def in_range(pt, img_shape):
        if pt[0] >= 0 and pt[0] < img_shape[1] and pt[1] >= 0 and pt[
                1] <= img_shape[0]:
            return True
        else:
            return False

    left, top, right, bot = crop_bbox
    h_img, w_img = img_shape[:2]
    crop_width = right - left
    crop_height = bot - top
    ratio_x = crop_width / w_img
    ratio_y = crop_height / h_img
    offset_x = (tgt_shape[1] - crop_width) / 2
    offset_y = top

    results = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            for pt in lanes[key]['points']:
                pt[0] = float(pt[0] * ratio_x + offset_x)
                pt[1] = float(pt[1] * ratio_y + offset_y)
                pts.append(pt)
            if len(pts) > 1:
                results.append(pts)
    return results

def vis_one(results, img, width=9,mode='gt_on'):
    # 增加了mode 表示不显示gt 
    img_gt=None
    img_pil = PIL.Image.fromarray(img)
    num_failed = 0
    
    def normalize_coords(coords):
        res = []
        for coord in coords:
            res.append((int(coord[0] + 0.5), int(coord[1] + 0.5)))
        return res
    preds = [normalize_coords(coord) for coord in results]
    
    for idx, pred_lane in enumerate(preds):
        PIL.ImageDraw.Draw(img_pil).line(
            xy=pred_lane, fill=COLORS[idx + 1], width=width)

    img = np.array(img_pil, dtype=np.uint8)


    return img, img_gt, num_failed

class CondNet():
    def __init__(self) -> None:
        args = parse_args()
        cfg = mmcv.Config.fromfile(args.config)
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        self.model = build_detector(cfg.model)
        if args.checkpoint is not None:
            load_checkpoint(self.model, args.checkpoint, map_location='cpu')
            
        self.model = self.model.cuda().eval()
        self.post_processor = CondLanePostProcessor(mask_size=(1, 40, 100), use_offset=True,hm_thr=0.5, nms_thr=4)
    
    def test(self,img):
        origin_img = img
        img = img[270:, ...]
        img = cv2.resize(img, SIZE)
        mean = np.array([75.3, 76.6, 77.6])
        std = np.array([50.5, 53.8, 54.3])
        img = mmcv.imnormalize(img, mean, std, False)
        x = torch.unsqueeze(torch.from_numpy(img).permute(2, 0, 1), 0)

        x = x.cuda()
        seeds, _ = self.model.test_inference(x)
        lanes, seeds = self.post_processor(seeds, 8)
        result = adjust_result(
                    lanes=lanes,
                    crop_bbox=[0, 270, 1640, 590],
                    img_shape=(320,800,3),
                    tgt_shape=(590, 1640))
        img_vis, img_gt_vis, num_failed = vis_one(result, origin_img,mode='gt_off')
        return img_vis

def callback(data):
    global my_condnet
    # Used to convert between ROS and OpenCV images
    br = CvBridge()
    
    # Output debugging information to the terminal
    rospy.loginfo("receiving video frame")
    
    # Convert ROS Image message to OpenCV image
    current_frame = br.imgmsg_to_cv2(data)
    result_frame = my_condnet.test(current_frame)
    # Display image
    rospy.loginfo('publishing video frame')
    pub.publish(br.cv2_to_imgmsg(result_frame))

def receive_message():
    global pub#不知道这样写可不可以实现pub共享
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are added to the end of the name. 
    rospy.init_node('CondNet_py', anonymous=True)

    # Node is subscribing to the video_frames topic
    rospy.Subscriber('video_frames', Image, callback)
    
    pub = rospy.Publisher('CondNetOutput', Image, queue_size=10)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    
    # Close down the video stream when done
    cv2.destroyAllWindows()
if __name__ == '__main__':
    my_condnet = CondNet()
    receive_message()
