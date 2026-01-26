import cv2
import numpy as np

def resize_frame_smart(frame, target_width=640):
    """
    Resizes a frame to a target width while maintaining aspect ratio,
    only if the frame is strictly larger than the target width.
    """
    height, width = frame.shape[:2]
    if width > target_width:
        scale_ratio = target_width / width
        new_width = target_width
        new_height = int(height * scale_ratio)
        return cv2.resize(frame, (new_width, new_height))
    return frame

def get_video_properties(cap):
    """
    Returns (fps, width, height, total_frames) of the video capture.
    """
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, total_frames

def create_video_writer(output_path, fps, width, height):
    """
    Creates a VideoWriter object with avc1 (H.264) codec.
    This is much more compatible with web browsers than mp4v.
    """
    # Try avc1 first, fallback to mp4v if not available
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
             raise Exception("avc1 not supported")
        return writer
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
