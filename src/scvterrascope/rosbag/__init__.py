"""ROS 2 bag reading utilities — wraps the `rosbags` pure-Python package.

Lets the GUI iterate `sensor_msgs/Image` (and CompressedImage) frames
without sourcing the system ROS environment. See `reader.BagReader`.
"""

from scvterrascope.rosbag.reader import BagReader, ImageTopicInfo
from scvterrascope.rosbag.types import BagFrame

__all__ = ["BagFrame", "BagReader", "ImageTopicInfo"]
