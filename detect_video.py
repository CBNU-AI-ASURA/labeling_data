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
                pose_result = pose.process(rgb_frame)  # 전역 pose 객체 사용
                hands_result = hands.process(rgb_frame)  # 전역 hands 객체 사용

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
        video_basename = os.path.basename(video_path).split('.')[0]
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


def setup_directories(process_num):
    base_dirs = {
        'video_root': 'raw/',
        'csv_root': 'landmarks_csv/',
        'output_root': 'labeled_csv/'
    }
    paths = {
        'videos_dir': os.path.join(base_dirs['video_root'], process_num),
        'csv_dir': os.path.join(base_dirs['csv_root'], process_num),
        'output_dir': os.path.join(base_dirs['output_root'], process_num)
    }

    for key, path in paths.items():
        if key != 'videos_dir':  # 입력 디렉토리는 생성하지 않음
            os.makedirs(path, exist_ok=True)

    return paths



def load_incomplete_files(file_path):
    """
    불완전한 파일 목록 로드.
    
    Parameters:
        file_path (str): 파일 목록 경로.

    Returns:
        set: 파일이 존재하면 해당 파일 목록을 반환, 존재하지 않으면 빈 세트를 반환.
    """
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist. Processing all files instead.")
        return set()  # 빈 세트 반환해 전체 비디오 파일 처리 가능
    
    with open(file_path, 'r') as f:
        return {
            os.path.basename(line.strip()).replace('landmarks_', '').replace('.csv', '.mp4') for line in f
        }


def create_args_list(videos_dir, csv_dir, morpheme_dir, output_dir, incomplete_files):
    """
    처리할 파일 목록 생성.

    Parameters:
        videos_dir (str): 입력 비디오 디렉토리.
        csv_dir (str): 추출된 CSV 저장 디렉토리.
        morpheme_dir (str): Morpheme JSON 디렉토리.
        output_dir (str): 라벨링된 CSV 저장 디렉토리.
        incomplete_files (set): 다시 처리할 파일 목록.

    Returns:
        list: 처리할 작업 목록.
    """
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    if not incomplete_files:  # 불완전 파일 목록이 비어 있으면 모든 파일 처리
        print("No incomplete files provided. Processing all available video files.")
        incomplete_files = set(video_files)

    args_list = [
        (os.path.join(videos_dir, video), csv_dir, morpheme_dir, output_dir)
        for video in video_files
        if video in incomplete_files
    ]

    print(f"Total matching files: {len(args_list)}")
    return args_list


def init_mediapipe():
    """
    각 프로세스가 Mediapipe 객체를 독립적으로 초기화하도록 설정.
    """
    global pose, hands
    pose = mp.solutions.pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def load_incomplete_files(file_path):
    """
    불완전한 파일 목록을 선택적으로 로드.

    Parameters:
        file_path (str): 파일 목록 경로.

    Returns:
        list: 선택된 파일 목록.
    """
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist.")
        return []

    # 파일 목록 읽기
    with open(file_path, 'r') as f:
        all_files = [os.path.basename(line.strip()) for line in f]

    # 선택적으로 파일 선택
    print("Incomplete files:")
    for idx, file in enumerate(all_files, start=1):
        print(f"{idx}. {file}")

    print("\nOptions:")
    print("1. Select specific files by index (comma-separated, e.g., 1,3,5)")
    print("2. Select all files")
    print("3. Cancel and exit")

    while True:
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            selected_indices = input("Enter indices of files to process: ").strip()
            try:
                selected_indices = [int(idx) for idx in selected_indices.split(',')]
                selected_files = [all_files[idx - 1] for idx in selected_indices]
                print(f"Selected files: {selected_files}")
                return selected_files
            except (ValueError, IndexError):
                print("Invalid input. Please enter valid indices.")
        elif choice == '2':
            print("Processing all files.")
            return all_files
        elif choice == '3':
            print("Exiting without processing.")
            return []
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == '__main__':
    while True:
        process_num = input("Enter process_num to process (or 'exit' to quit): ").strip()
        if process_num.lower() == 'exit':
            print("Exiting...")
            break

        incomplete_files_path = f'incomplete_files_{process_num}.txt'
        morpheme_dir = 'morpheme/'

        print(f"Starting processing for process_num: {process_num}")

        # 디렉토리 설정
        paths = setup_directories(process_num)
        videos_dir = paths['videos_dir']
        csv_dir = paths['csv_dir']
        output_dir = paths['output_dir']

        # 불완전 파일 목록 로드 (없으면 모든 파일 처리)
        incomplete_files = load_incomplete_files(incomplete_files_path)

        # 처리할 작업 목록 생성
        args_list = create_args_list(videos_dir, csv_dir, morpheme_dir, output_dir, incomplete_files)

        total = len(args_list)
        print(f"Processing {total} videos for process_num {process_num}")

        if total == 0:
            print(f"No videos to process for process_num {process_num}. Skipping...")
            continue

        # 멀티프로세싱 작업 시작
        try:
            with Pool(processes=10, initializer=init_mediapipe) as pool:
                for _ in tqdm(pool.imap_unordered(process_video_and_map, args_list), total=total, desc=f'Processing {process_num}'):
                    pass
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt detected for process_num {process_num}, terminating pool...")
            pool.terminate()
            pool.join()
            break

        print(f"Completed processing for process_num: {process_num}.")