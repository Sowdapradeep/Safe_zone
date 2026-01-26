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
    Tries multiple codecs to find one supported by the system.
    """
    # Prefer avc1/h264 for web compatibility
    codecs = ['avc1', 'h264', 'H264', 'x264', 'X264', 'mp4v']
    
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"Succesfully opened VideoWriter with codec: {codec}")
                return writer
        except Exception as e:
            print(f"Codec {codec} failed: {e}")
            continue
            
    # Final fallback attempt
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
