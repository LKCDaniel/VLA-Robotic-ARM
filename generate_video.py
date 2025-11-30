import os
import cv2
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


def main():
    # for episode_index in trange(400):
    # episode_save_dir = os.path.join(os.path.dirname(__file__), "episodes", f"episode_{episode_index}")
    episode_save_dir = os.path.join(os.path.dirname(__file__), "real_time_test")
    camera_1_dir = os.path.join(episode_save_dir, "camera_1")
    video_path_1 = os.path.join(episode_save_dir, "camera_1.mp4")
    images_to_video(camera_1_dir, video_path_1)

    camera_2_dir = os.path.join(episode_save_dir, "camera_2")
    video_path_2 = os.path.join(episode_save_dir, "camera_2.mp4")
    images_to_video(camera_2_dir, video_path_2)

    camera_3_dir = os.path.join(episode_save_dir, "camera_3")
    video_path_3 = os.path.join(episode_save_dir, "camera_3.mp4")
    images_to_video(camera_3_dir, video_path_3)


if __name__ == "__main__":
    main()