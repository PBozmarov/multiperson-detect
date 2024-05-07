import numpy as np
from tqdm import tqdm
import natsort
import argparse
import sys
import os
import pandas as pd
from ultralytics import YOLO
import cv2
from time import time

# Path to the folder containing test.py of our anti-spoof repository
repo_path = os.path.abspath("./anti_spoof")
sys.path.append(repo_path)

from spoof_test import test as spoof_test


def test_tuned(
    video_path: str,
    skip_frames: int = 1,
    verbose: bool = False,
    show_video: bool = False,
):

    """
    Function to test the fine-tuned yolov8n.pt model on our custom images dataset
    Args:
        video_path: str, path to the video file
        skip_frames: int, number of frames to skip before processing the next frame, speeds up the processing
        verbose: bool, prints the object detection results
        show_video: bool, shows the video with the object detection results
    Returns:
        1 if two or more people are detected in the same frame, 0 otherwise
    """
    if verbose:
        start = time()

    # Load the fine-tuned Yolo V8 model
    model_path = "trained_models/yolov8_n_pt_epochs_100_sz_640/weights/best.pt"
    model = YOLO(model_path)    

    if verbose:
        print(model.names)

    video_capture = cv2.VideoCapture(video_path)

    # get the frames per second
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    ret = True
    num_cons_frames_more_than_1_person = 0
    frame_counter = 0
    target_consecutive_frames = int((0.6 * fps) / (skip_frames + 1))

    while ret:

        # skip "skip_frames" number of frames
        ret, frame = video_capture.read()

        if ret and frame_counter % (skip_frames + 1) == 0:

            # detect the objects in the frame
            results = model.predict(frame, verbose=verbose, conf=0.55)
            real_results = results[0]
            boxes = real_results.boxes.numpy()

            num_real_people_in_frame = 0

            # Loop through the detected objects
            for box in boxes:

                conf = box.conf[0]
                class_id = box.cls[0]
                class_label = model.names[class_id]

                # if the id corresponds to a real person, increment the count
                if class_id == 0:
                        num_real_people_in_frame += 1

                if show_video:

                    x1 = int(box.xyxy[0][0])
                    y1 = int(box.xyxy[0][1])
                    x2 = int(box.xyxy[0][2])
                    y2 = int(box.xyxy[0][3])
                    bbox = (x1, y1, x2, y2)
                    bbox_color = (
                        (0, 0, 255) if class_label == "real" else (255, 0, 0)
                    )  # Red for 'real', Blue for 'fake'
                    text_color = (0, 0, 255) if class_label == "real" else (255, 0, 0)
                    line_thickness = 3
                    font_scale = 1.2

                    # Draw the bounding box
                    cv2.rectangle(
                        frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        bbox_color,
                        line_thickness,
                    )
                    cv2.putText(
                        frame,
                        f"{class_label} {conf:.2f}",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        text_color,
                        line_thickness,
                    )

            # if there are more than 1 real person in the frame, increment the count
            if num_real_people_in_frame > 1:
                num_cons_frames_more_than_1_person += 1
            else:
                num_cons_frames_more_than_1_person = 0

            # if the count of consecutive frames with more than 1 person is greater than the target, return 1
            if num_cons_frames_more_than_1_person >= target_consecutive_frames:

                if show_video:
                    text = "Two or more people detected!"
                    location = (50, 50)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_color = (0, 255, 0)
                    line_type = 2
                    cv2.putText(
                        frame, text, location, font, font_scale, font_color, line_type
                    )

                return 1

            if show_video:

                # visualize the frame
                cv2.imshow("frame", frame)

            # Pause the video when 'p' is pressed
            key = cv2.waitKey(1)
            if key & 0xFF == ord("p"):
                cv2.waitKey()  # Wait indefinitely for the user to press any key

            if key & 0xFF == ord("q"):
                break

        frame_counter += 1

    else:

        # if no frame has two or more people, return 0
        return 0

    if show_video:
        video_capture.release()
        cv2.destroyAllWindows()

    if verbose:
        end = time()
        print(f"Time taken: {end - start} seconds")


def test_pre_trained(
    video_path: str,
    skip_frames: int = 1,
    verbose: bool = False,
    show_video: bool = False,
):

    """
    Function to test on a video the yolov8npt model, pretrained on the COCO dataset
    Args:
        video_path: str, path to the video file
        skip_frames: int, number of frames to skip before processing the next frame, speeds up the processing
        verbose: bool, prints the object detection results
        show_video: bool, shows the video with the object detection results
    Returns:
        1 if two or more people are detected in the same frame, 0 otherwise
    """

    if verbose:
        start = time()

    # Load Yolo V8 pre-trained model
    model = YOLO("yolov8n.pt")
    if verbose:
        print(model.names)

    video_capture = cv2.VideoCapture(video_path)

    # get the frames per second
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    ret = True
    num_cons_frames_more_than_1_person = 0
    frame_counter = 0
    target_consecutive_frames = int((0.6 * fps) / (skip_frames + 1))

    while ret:

        ret, frame = video_capture.read()
        
        # skip "skip_frames" number of frames
        if ret and frame_counter % (skip_frames + 1) == 0:

            # detect and track objects in the frame
            results = model.predict(frame, verbose=verbose, classes=0, conf=0.55)
            real_results = results[0]
            boxes = real_results.boxes.numpy()

            num_real_people_in_frame = 0
            for box in boxes:

                conf = box.conf[0]
                class_id = box.cls[0]
                class_label = model.names[class_id]
                conf = box.conf[0]
                x1 = int(box.xyxy[0][0])
                y1 = int(box.xyxy[0][1])
                x2 = int(box.xyxy[0][2])
                y2 = int(box.xyxy[0][3])
                bbox = (x1, y1, x2, y2)
                roi = frame[y1:y2, x1:x2]
                is_real, is_real_score = spoof_test(roi)
                if is_real_score < 0.6:
                    continue
                is_real = is_real == 1
                if is_real:
                    num_real_people_in_frame += 1

                if show_video:
                    class_id = box.cls[0]
                    class_label = model.names[class_id]
                    class_label = "real person" if is_real else "fake person"
                    bbox_color = (
                        (0, 0, 255) if is_real else (255, 0, 0)
                    )  # Red for 'real', Blue for 'fake'
                    text_color = (
                        (0, 0, 255) if is_real else (255, 0, 0)
                    )  
                    line_thickness = 3  
                    font_scale = 1.2  

                    # Draw the bounding box
                    cv2.rectangle(
                        frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        bbox_color,
                        line_thickness,
                    )

                    # Put the label and confidence score on the frame above the top-left corner of the bounding box
                    cv2.putText(
                        frame,
                        f"{class_label} {conf:.2f}",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        text_color,
                        line_thickness,
                    )

            if num_real_people_in_frame > 1:
                num_cons_frames_more_than_1_person += 1
            else:
                num_cons_frames_more_than_1_person = 0

            if num_cons_frames_more_than_1_person >= target_consecutive_frames:

                if show_video:
                    text = "Two or more people detected!"
                    location = (50, 50)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_color = (0, 255, 0)
                    line_type = 2
                    cv2.putText(
                        frame, text, location, font, font_scale, font_color, line_type
                    )

                    time.sleep(5)  # Wait for 5 seconds

                return 1

            if show_video:
                # visualize the frame
                cv2.imshow("frame", frame)

            # Pause the video when 'p' is pressed
            key = cv2.waitKey(1)
            if key & 0xFF == ord("p"):
                cv2.waitKey()  # Wait indefinitely for the user to press any key

            if key & 0xFF == ord("q"):
                break

        frame_counter += 1

    else:
        return 0

    if show_video:
        video_capture.release()
        cv2.destroyAllWindows()

    if verbose:
        end = time()
        print(f"Time taken: {end - start} seconds")


def test_folder(folder_path: str, model_type: str = "tuned", skip_frames: int = 1):
    """
    Function to test all videos in a folder using the fine-tuned or pre-trained Yolo V8 model.
    Args:
        folder_path: str, path to the folder containing the videos
        model_type: str, 'tuned'- fine-tuned yolov8.pt on our custom images or the standard 'pre_trained' yolov8.pt on COCO dataset,
        skip_frames: int, number of frames to skip before processing the next frame, speeds up the processing
    Returns:
        results: list of tuples, each tuple contains the prediction and the video name
    """
    videos = os.listdir(folder_path)
    videos = [file for file in videos if file.split(".")[-1] in ["mp4", "avi", "mov"]]
    videos = natsort.natsorted(videos)
    results = []

    with open(os.path.join(folder_path, "predictions.txt"), "w") as file:
        file.write("Prediction     Label\n")
        for video in tqdm(videos, desc="Processing videos"):
            video_name = video.split(".")[0]
            video_path = os.path.join(folder_path, video)
            if model_type == "tuned":
                result = test_tuned(
                    video_path, skip_frames, verbose=False, show_video=False
                )
            else:
                result = test_pre_trained(
                    video_path, skip_frames, verbose=False, show_video=False
                )

            # Write the result immediately after getting it
            file.write(f"{result}             {video_name}\n")
            file.flush()

            results.append((result, video_name))

    print(f"Results saved at {os.path.join(folder_path, 'predictions.txt')}")

    return results


if __name__ == "__main__":

    desc = "Detect multiple people at the same time in a video."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "folder"],
        default="single",
        help="single: test a single video, folder: test all videos in a folder",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["tuned", "pre_trained"],
        default="tuned",
        help="tuned: fined-tuned yolov8n.pt, pre_trained: yolov8n.pt",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        help="absolute or relative path to the video file or a video folder (depending on the mode)",
    )
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=1,
        help="number of frames to skip before processing the next frame, speeds up the processing",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="prints the object detection results",
    )
    parser.add_argument(
        "--show_video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="shows the video with the object detection results",
    )

    args = parser.parse_args()
    if args.mode == "single":
        start = time()
        if args.model_type == "tuned":
            result = test_tuned(
                args.video_path, args.skip_frames, args.verbose, args.show_video
            )
        else:
            result = test_pre_trained(
                args.video_path, args.skip_frames, args.verbose, args.show_video
            )
        if result == 1:
            print(f"Multiple people detected at the same moment!")
        else:
            print("NO multiple people detected at the same moment!")
        end = time()
        print(f"Time taken: {end - start} seconds")

    else:
        test_folder(args.video_path, args.model_type, args.skip_frames)
