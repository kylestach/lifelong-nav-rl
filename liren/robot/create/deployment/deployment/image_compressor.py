import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2

import numpy as np
import tensorflow as tf

class ImageCompressor(Node):
    def __init__(self):
        super().__init__('image_compressor')
        self.subscription = self.create_subscription(
            Image,
            '/front/image_raw',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(
            CompressedImage,
            '/front/image_raw/compressed_rgb',
            10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        success, compressed_image = cv2.imencode('.jpg', cv_image)
        if not success:
            self.get_logger().error('Failed to compress image')
            return
        compressed_image_msg = CompressedImage()
        compressed_image_msg.header = msg.header
        compressed_image_msg.format = "jpeg"
        compressed_image_msg.data = compressed_image.tobytes()

        self.publisher.publish(compressed_image_msg)

if __name__ == '__main__':
    rclpy.init()
    image_compressor = ImageCompressor()
    rclpy.spin(image_compressor)
