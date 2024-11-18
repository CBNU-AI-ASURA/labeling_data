import os
import sys
from multiprocessing import Pool
from tqdm import tqdm
import mediapipe as mp
import cv2
import csv
import json
import pandas as pd

# TensorFlow 및 Mediapipe 경고 로그 무시
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['KMP_WARNINGS'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'


def process_video_and_map(args):
    sys.stderr.flush()
    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), 2)

    video_path, csv_dir, morpheme_dir, output_dir = args
    try:
        # MediaPipe Pose와 Hands 초기화
        mp_hands = mp.solutions.hands
        pose = mp.solutions.pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 비디오 처리
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        # CSV 파일 생성
        csv_filename = f'landmarks_{os.path.basename(video_path).split(".")[0]}.csv'
        csv_filepath = os.path.join(csv_dir, csv_filename)

        with open(csv_filepath, mode='w', newline='') as landmarks_file:
            csv_writer = csv.writer(landmarks_file)
            csv_writer.writerow(['frame_num', 'landmark_type', 'index', 'x', 'y', 'z'])
            frame_num = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = pose.process(rgb_frame)
                hands_result = hands.process(rgb_frame)

                if pose_result.pose_landmarks:
                    for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
                        csv_writer.writerow([frame_num, 'pose', idx, landmark.x, landmark.y, landmark.z])

                if hands_result.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            hand_label = hands_result.multi_handedness[hand_idx].classification[0].label
                            hand_type = 'left_hand' if hand_label == 'Left' else 'right_hand'
                            csv_writer.writerow([frame_num, hand_type, idx, landmark.x, landmark.y, landmark.z])

        cap.release()

        # Morpheme 매핑
        # ex) NIA_SL_SEN0001_REAL01_D_morpheme.json
        video_basename = os.path.basename(video_path).split('.')[0] # NIA_SL_SEN0001_REAL01_D_morpheme
        participant_num = video_basename.split('_')[3][-2:]
        morpheme_file = os.path.join(morpheme_dir, participant_num, f"{video_basename}_morpheme.json")

        if not os.path.exists(morpheme_file):
            print(f"Morpheme file not found: {morpheme_file}")
            return

        with open(morpheme_file, 'r', encoding='utf-8') as f:
            morpheme_data = json.load(f)

        keypoint_df = pd.read_csv(csv_filepath)
        keypoint_df['time'] = keypoint_df['frame_num'] / 30.0  # FPS = 30

        labeled_frames = []
        for segment in morpheme_data['data']:
            start_time = segment['start']
            end_time = segment['end']
            label = segment['attributes'][0]['name']

            segment_df = keypoint_df[
                (keypoint_df['time'] >= start_time) & (keypoint_df['time'] <= end_time)
            ].copy()
            segment_df['label'] = label
            labeled_frames.append(segment_df)

        if labeled_frames:
            labeled_keypoint_df = pd.concat(labeled_frames)
            output_csv_filename = f"labeled_{os.path.basename(csv_filepath)}"
            output_csv_path = os.path.join(output_dir, output_csv_filename)
            labeled_keypoint_df.to_csv(output_csv_path, index=False)

    except Exception as e:
        print(f"Error processing {video_path}: {e}", file=sys.stderr)


if __name__ == '__main__':
    
    process_num = '01'

    video_root = 'raw/'
    csv_root = 'landmarks_csv/'
    output_root = 'labeled_csv/'
    videos_dir = os.path.join(video_root, process_num)
    csv_dir = os.path.join(csv_root, process_num)
    output_dir = os.path.join(output_root, process_num)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    morpheme_dir = 'morpheme/'

    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]

    processed_files = {
        f.split('_')[1].split('.')[0]
        for f in os.listdir(output_dir)
        if f.startswith('labeled_') and f.endswith('.csv')
    }

    args_list = [
        (os.path.join(videos_dir, video), csv_dir, morpheme_dir, output_dir)
        for video in video_files
        if video.split('.')[0] not in processed_files
    ]

    total = len(args_list)
    print(f"Processing {total} / {len(video_files)} videos")

    try:
        with Pool(processes=20) as pool:
            for _ in tqdm(pool.imap_unordered(process_video_and_map, args_list), total=total, desc='Process videos'):
                pass
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, terminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)