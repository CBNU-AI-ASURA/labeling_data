import os
import pandas as pd
from glob import glob
from tqdm import tqdm


def find_incomplete_csv_files(csv_dir, output_incomplete_file):
    """
    동일 문장의 두 방향 이상의 CSV 파일 중 길이 차이가 70% 이상 나거나,
    헤더만 있거나 비어 있는 파일을 식별.

    Parameters:
        csv_dir (str): CSV 파일이 저장된 디렉토리 경로.
        output_incomplete_file (str): 중간에 끊겨 있는 파일 목록 저장 경로.
    """
    csv_files = glob(os.path.join(csv_dir, '*.csv'))
    file_lengths = {}
    incomplete_files = []

    # 파일별 길이 계산 및 비어 있는 파일 확인
    for csv_file in tqdm(csv_files, desc="Checking file lengths"):
        try:
            df = pd.read_csv(csv_file)
            file_name = os.path.basename(csv_file)
            base_name, _ = os.path.splitext(file_name)  # 확장자 제거
            parts = base_name.split('_')

            # 파일명 형식 확인 및 문장 ID와 방향 추출
            if len(parts) < 6:
                raise ValueError(f"Invalid file format: {file_name}")
            sentence_id = parts[3]  # SENXXXX
            direction = parts[-1]  # D, F, L, R, U

            # 파일이 비어 있는 경우
            if df.empty:
                incomplete_files.append(csv_file)
                continue

            # 문장 ID별 파일 길이 저장
            if sentence_id not in file_lengths:
                file_lengths[sentence_id] = {}
            file_lengths[sentence_id][direction] = len(df)
        except Exception as e:
            tqdm.write(f"Error reading {csv_file}: {e}")
            incomplete_files.append(csv_file)

    # 길이 차이 검증
    for sentence_id, directions in tqdm(file_lengths.items(), desc="Validating lengths"):
        lengths = list(directions.values())
        if len(lengths) >= 2:  # 두 방향 이상 파일이 있는 경우만 검증
            max_length = max(lengths)
            for direction, length in directions.items():
                if max_length > 0 and length / max_length < 0.7:  # 70% 미만 길이
                    file_name = f"landmarks_NIA_SL_{sentence_id}_REAL01_{direction}.csv"
                    incomplete_files.append(os.path.join(csv_dir, file_name))

    # 결과 저장
    with open(output_incomplete_file, 'w') as f:
        for file in set(incomplete_files):  # 중복 제거
            f.write(file + '\n')

    print(f"Found {len(set(incomplete_files))} incomplete files. Saved to {output_incomplete_file}")


if __name__ == "__main__":
    csv_dir = 'landmarks_csv/01/'  # CSV 파일 디렉토리
    output_incomplete_file = 'incomplete_files.txt'  # 중간에 끊겨 있는 파일 목록

    # 중간에 끊겨 있는 CSV 파일 찾기
    find_incomplete_csv_files(csv_dir, output_incomplete_file)