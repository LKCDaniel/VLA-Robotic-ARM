import os
import cv2
import argparse
from tqdm import trange


def images_to_video(image_dir, output_path):
    images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
    num_images = len(images)
    images = [f"{index}.png" for index in range(num_images)]

    first = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = first.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # 或者 'H264', 'X264', 'avc1'
    # fourcc = cv2.VideoWriter_fourcc(*'H264')  # 有些系统更好
    fps = 30
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()


def main(episode_from, episode_to, is_real_time_test=False):
    if is_real_time_test:
        episode_image_dir = os.path.join(os.path.dirname(__file__), "real_time_test")
        episode_video_dir = os.path.join(os.path.dirname(__file__), "real_time_test", "videos")
        os.makedirs(episode_video_dir, exist_ok=True)
        
        camera_1_dir = os.path.join(episode_image_dir, "camera_1")
        video_path_1 = os.path.join(episode_video_dir, "camera_1.mp4")
        images_to_video(camera_1_dir, video_path_1)

        camera_2_dir = os.path.join(episode_image_dir, "camera_2")
        video_path_2 = os.path.join(episode_video_dir, "camera_2.mp4")
        images_to_video(camera_2_dir, video_path_2)

        camera_3_dir = os.path.join(episode_image_dir, "camera_3")
        video_path_3 = os.path.join(episode_video_dir, "camera_3.mp4")
        images_to_video(camera_3_dir, video_path_3)
        
        return
    
    if episode_to == -1:
        episode_to = len(os.listdir(os.path.join(os.path.dirname(__file__), "episodes")))
        
    for episode_index in trange(episode_from, episode_to):
        episode_image_dir = os.path.join(os.path.dirname(__file__), "episodes", f"episode_{episode_index}")
        episode_video_dir = os.path.join(os.path.dirname(__file__), "episodes", "videos", f"episode_{episode_index}")
        os.makedirs(episode_video_dir, exist_ok=True)
        
        # episode_image_dir = os.path.join(os.path.dirname(__file__), "real_time_test")
        camera_1_dir = os.path.join(episode_image_dir, "camera_1")
        video_path_1 = os.path.join(episode_video_dir, "camera_1.mp4")
        images_to_video(camera_1_dir, video_path_1)

        camera_2_dir = os.path.join(episode_image_dir, "camera_2")
        video_path_2 = os.path.join(episode_video_dir, "camera_2.mp4")
        images_to_video(camera_2_dir, video_path_2)

        camera_3_dir = os.path.join(episode_image_dir, "camera_3")
        video_path_3 = os.path.join(episode_video_dir, "camera_3.mp4")
        images_to_video(camera_3_dir, video_path_3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate videos from images")
    parser.add_argument("--episode_from", type=int, required=False, default=0)
    parser.add_argument("--episode_to", type=int, required=False, default=-1)
    parser.add_argument("--is_real_time_test", action='store_true', help="whether to process real time test images")
    args = parser.parse_args()
    
    main(args.episode_from, args.episode_to, args.is_real_time_test)
    
    # python ./generate_video.py --episode_from 1 --episode_to 2