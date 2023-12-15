import cv2
import os


class VideoFrameExtractor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def extract_frames_to_folder(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for idx in range(self.total_frames):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.video.read()
            if ret:
                frame_path = os.path.join(output_folder, f"frame_{idx}.jpg")
                cv2.imwrite(frame_path, frame)

    def release(self):
        self.video.release()


# 使用示例
video_path = 'example001.mp4'  # 替换为你的视频路径
output_frames_folder = 'v1_frame'  # 输出帧的文件夹路径
if not os.path.exists(output_frames_folder):
    os.makedirs(output_frames_folder)
frame_extractor = VideoFrameExtractor(video_path)
frame_extractor.extract_frames_to_folder(output_frames_folder)
frame_extractor.release()
