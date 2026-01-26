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
    Creates a VideoWriter object with a browser-compatible codec.
    Prefers WebM/VP8 for universal browser support.
    """
    # 1. Try WebM/VP8 (Very safe for browsers)
    try:
        if output_path.endswith('.mp4'):
            output_path = output_path.replace('.mp4', '.webm')
            
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"Succesfully opened VideoWriter with WebM/VP8 (VP80)")
            return writer, output_path
    except:
        pass

    # 2. Try H.264 (avc1)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"Succesfully opened VideoWriter with H.264 (avc1)")
            return writer, output_path
    except:
        pass
            
    # 3. Final fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height)), output_path
