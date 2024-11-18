import os
import pandas as pd
import json
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager


# Step 1: 손상된 CSV 파일 식별
def check_csv_files(csv_dir, output_list_path):
    """
    CSV 파일의 무결성을 검사하고 손상된 파일 리스트를 출력합니다.

    Parameters:
        csv_dir (str): CSV 파일이 저장된 디렉토리 경로
        output_list_path (str): 손상된 파일 리스트를 저장할 경로
    """
    csv_files = glob(os.path.join(csv_dir, '*.csv'))
    print(f"Found {len(csv_files)} files to check.")
    damaged_files = []

    for csv_file in tqdm(csv_files, desc="Checking CSV files"):
        try:
            # CSV 파일 로드
            df = pd.read_csv(csv_file)

            # 비어 있는 파일 또는 특정 컬럼이 없는 경우 추가
            if df.empty or 'frame_num' not in df.columns:
                damaged_files.append(csv_file)
        except Exception as e:
            # 파일 로드 실패한 경우 추가
            damaged_files.append(csv_file)

    # 손상된 파일 리스트 저장
    with open(output_list_path, 'w') as f:
        for file in damaged_files:
            f.write(file + '\n')

    print(f"Check completed. Damaged files: {len(damaged_files)} (saved in {output_list_path})")


# Step 2: Morpheme 매핑 및 라벨링
def mapping(csv_file, progress_counter, failed_files):
    try:
        # 파일명에서 정보 추출
        filename = os.path.basename(csv_file)
        parts = filename.split('_')
        sentence_id = parts[1]
        video_info = '_'.join(parts[1:]).split('.')[0]
        video_parts = video_info.split('_')
        participant_info = video_parts[3]
        participant_num = participant_info[-2:]

        # 키포인트 데이터 로드
        keypoint_df = pd.read_csv(csv_file)
        keypoint_df['time'] = keypoint_df['frame_num'] / FPS

        # 해당 morpheme JSON 파일 로드
        morpheme_filename = f"{video_info}_morpheme.json"
        morpheme_file = os.path.join(morpheme_dir, participant_num, morpheme_filename)

        if not os.path.exists(morpheme_file):
            raise FileNotFoundError(f"Morpheme file not found: {morpheme_filename}")

        with open(morpheme_file, 'r', encoding='utf-8') as f:
            morpheme_data = json.load(f)

        segments = morpheme_data['data']
        labeled_frames = []

        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            label = segment['attributes'][0]['name']

            segment_df = keypoint_df[
                (keypoint_df['time'] >= start_time) &
                (keypoint_df['time'] <= end_time)
            ].copy()
            segment_df['label'] = label
            labeled_frames.append(segment_df)

        if labeled_frames:
            labeled_keypoint_df = pd.concat(labeled_frames)
            output_csv_filename = f"labeled_{filename}"
            output_csv_path = os.path.join(output_dir, output_csv_filename)
            labeled_keypoint_df.to_csv(output_csv_path, index=False)
    except Exception as e:
        failed_files.append(csv_file)
        tqdm.write(f"Error processing {csv_file}: {e}")
    finally:
        progress_counter.value += 1


def process_all_files(csv_dir, output_dir, damaged_files_path):
    """
    올바른 CSV 파일을 대상으로 Morpheme 매핑 수행.

    Parameters:
        csv_dir (str): CSV 파일이 저장된 디렉토리 경로
        output_dir (str): 라벨링된 CSV 파일이 저장될 경로
        damaged_files_path (str): 손상된 파일 리스트 경로
    """
    csv_files = glob(os.path.join(csv_dir, '*.csv'))

    # 손상된 파일 제외
    with open(damaged_files_path, 'r') as f:
        damaged_files = set(f.read().splitlines())
    valid_files = [file for file in csv_files if file not in damaged_files]

    print(f"Found {len(valid_files)} valid files to process.")

    manager = Manager()
    progress_counter = manager.Value('i', 0)
    failed_files = manager.list()

    with tqdm(total=len(valid_files), desc="Processing files", position=0, leave=True) as pbar:
        num_workers = min(cpu_count(), len(valid_files))
        with Pool(num_workers) as pool:
            for csv_file in valid_files:
                pool.apply_async(
                    mapping,
                    args=(csv_file, progress_counter, failed_files),
                    callback=lambda _: pbar.update(1)
                )
            pool.close()
            pool.join()

    failed_files_path = os.path.join(output_dir, 'failed_files.txt')
    with open(failed_files_path, 'w') as f:
        for file in failed_files:
            f.write(file + '\n')

    print(f"\nProcessing completed. Failed files: {len(failed_files)} (saved in {failed_files_path})")


if __name__ == "__main__":
    # 경로 설정
    csv_dir = 'landmarks_csv/'
    output_dir = 'labeled_csv/'
    morpheme_dir = 'morpheme/'
    damaged_files_path = 'damaged_files.txt'
    FPS = 30

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: CSV 무결성 검사
    check_csv_files(csv_dir, damaged_files_path)

    # Step 2: Morpheme 매핑
    process_all_files(csv_dir, output_dir, damaged_files_path)