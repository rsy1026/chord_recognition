import json
import re
import glob
import numpy as np
import pandas as pd
from typing import Optional

from chord.constants import IDX_TO_PITCH_CLASS, QUALITY_TO_IDX, IDX_TO_QUALITY

# 12 반음계 및 7 Quality 맵핑 (이전과 동일)
PITCH_MAP = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
INTERVAL_MAP = {
    "1": 0,
    "b2": 1,
    "2": 2,
    "b3": 3,
    "3": 4,
    "4": 5,
    "#4": 6,
    "b5": 6,
    "5": 7,
    "#5": 8,
    "b6": 8,
    "6": 9,
    "b7": 10,
    "7": 11,
}


def parse_choco_harte_label(harte_string: str) -> Optional[int]:
    """
    ChoCo의 기괴한 Harte 문자열을 정제하여 (Root, Quality, Bass) 인덱스로 압축합니다.
    """
    if harte_string in ["N", "Z", "X", "None"] or not harte_string:
        return None  # 묵음 구간

    # Harte 문법 분해 정규식
    # 예: 'C#:min(*5,b7)/b3'
    # -> Root: 'C#', Shorthand: 'min', Intervals: '(*5,b7)', Bass: 'b3'
    pattern = r"^([A-G][#b]?)(?::([a-zA-Z0-9]+))?(?:\([^)]+\))?(?:/([#b]?\d))?$"
    match = re.match(pattern, harte_string)

    # 1. 정규식에 안 잡히는 극단적인 구성음 나열식 표기 (예: C:(b3, 5, b7)) 방어
    if not match:
        fallback_match = re.match(
            r"^([A-G][#b]?):?\((.*?)\)(?:/([#b]?\d))?$", harte_string
        )
        if fallback_match:
            root_str, intervals, bass_str = fallback_match.groups()
            qual_str = (
                "min" if "b3" in intervals else "maj"
            )  # 대충 메이저/마이너만 가름
        else:
            return None  # 해석 불가 시 버림
    else:
        root_str, qual_str, bass_str = match.groups()

    root_idx = PITCH_MAP.get(root_str, 0)

    # 2. Quality 맵핑 (8 Quality 확장 체제)
    qual_str = qual_str.lower() if qual_str else "major"

    if qual_str in ["maj7", "maj9"]:
        qual_idx = "major-seventh"
    elif qual_str in ["min7", "min9", "m7"]:
        qual_idx = "minor-seventh"
    elif qual_str in ["hdim", "hdim7", "m7b5", "min7b5"]:
        qual_idx = "half-diminished"  # Half-diminished 부활
    elif qual_str in ["dim", "dim7"]:
        qual_idx = "diminished"
    elif qual_str in ["aug"]:
        qual_idx = "augmented"
    elif qual_str in ["7", "9", "11", "13", "dom", "dom7"]:
        qual_idx = "dominant-seventh"
    elif qual_str in ["min", "m"]:
        qual_idx = "minor"
    else:
        qual_idx = "major"  # maj, sus4, 파워코드 등

    qual_idx = QUALITY_TO_IDX[qual_idx]

    # 3. Bass 처리
    if bass_str and bass_str in INTERVAL_MAP:
        bass_interval = INTERVAL_MAP[bass_str]
        bass_idx = (root_idx + bass_interval) % 12
    else:
        bass_idx = root_idx

    # 인덱스 계산 공식도 8 Quality(96개 뼈대) 체제에 맞게 변경
    # Root(12) * 96 + Quality(8) * 12 + Bass(12)
    state_index = root_idx * 96 + qual_idx * 12 + bass_idx

    return state_index


def extract_sequences_from_choco(jams_folder_path: str) -> list:
    """
    ChoCo JAMS 파일들을 순회하며 R&B, Soul, Jazz, Funk 장르의 화음 시퀀스만 추출합니다.
    """
    sequences = []
    jams_files = glob.glob(f"{jams_folder_path}/*.jams", recursive=True)

    for file_path in jams_files:
        try:
            with open(file_path, "r") as f:
                jams_data = json.load(f)

            # 1. 장르 필터링 (메타데이터 활용)
            # JAMS 내부의 'sandbox' 또는 'file_metadata'를 뒤져 장르 확인
            metadata = jams_data["file_metadata"]
            artist = metadata.get("artist", "").lower()
            title = metadata.get("title", "").lower()

            # (주의: ChoCo는 원본 데이터셋마다 장르 태그 위치가 달라
            # 확실하게 R&B/Jazz만 쓸거면 McGill Billboard의 특정 하위 폴더만 타겟팅하는 것이 안전합니다)

            # 2. 화음 어노테이션 추출
            # 'chord' 또는 'chord_harte' 네임스페이스 탐색
            chord_anno = None
            for anno in jams_data["annotations"]:
                if anno["namespace"] in ["chord", "chord_harte"]:
                    chord_anno = anno["data"]
                    break

            if not chord_anno:
                continue

            # 3. 시퀀스 압축
            seq = []
            for obs in chord_anno:
                harte_label = obs["value"]
                state_idx = parse_choco_harte_label(harte_label)
                seq.append(state_idx)

            sequences.append(seq)

        except Exception as e:
            # 깨진 JAMS 파일은 과감히 패스
            continue

    return sequences


import numpy as np


def compute_chord_transition_matrix_numpy(
    chord_seqs: list,
    num_states: int = 1152,
    alpha: float = 0.1,
    return_count: bool = False,
) -> np.ndarray:
    """
    1. 분모 부풀림 버그 완벽 수정
    2. 라플라스 스무딩(alpha)을 통한 무한대(Inf) 발산 방지
    3. 고속 Viterbi 연산을 위한 Log-Transition Matrix 반환
    """
    # 전이 카운트를 담을 2D 행렬 초기화 (행: prev_chord, 열: curr_chord)
    # 정밀한 부동소수점 연산을 위해 float64 지정
    count_matrix = np.zeros((num_states, num_states), dtype=np.float64)

    # 1. 전이 카운트 매핑 (분모를 따로 세지 않고, 행렬의 행 단위 합으로 분모를 대체)
    for seq in chord_seqs:
        for t in range(len(seq) - 1):
            prev_state = seq[t]
            curr_state = seq[t + 1]

            if prev_state is not None and curr_state is not None:
                count_matrix[prev_state, curr_state] += 1

    if return_count:
        return count_matrix

    # 2. 라플라스 스무딩 적용
    # 데이터셋에 없는 전이도 alpha만큼의 가상의 카운트를 획득하여 안심번호 확보
    count_matrix += alpha

    # 3. 조건부 확률 계산: P(B|A) = Count(A->B) / Sum_over_all_X(Count(A->X))
    # 각 행(row)의 합이 곧 정확한 분모가 됨 (마지막 코드 카운트 오류 원천 차단)
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    transition_matrix = count_matrix / row_sums

    # 4. 언더플로우 방지 및 Viterbi 연산 최적화를 위한 로그 변환
    # 스무딩 덕분에 transition_matrix의 모든 원소는 > 0 이므로 안전하게 로그 연산 가능
    log_transition_matrix = np.log(transition_matrix)

    # 음의 로그 확률(-log P)이 필요하다면 아래 주석을 해제하되,
    # 통상적인 Viterbi는 그냥 log P를 더해나가는 맥스 풀링을 씁니다.
    # return -log_transition_matrix

    return log_transition_matrix


def decode_state_to_chord(state_idx: int) -> str:
    """
    1152차원 인덱스를 인간이 읽을 수 있는 화음 문자열(예: 'C:maj7/E')로 복원합니다.
    이전 인덱스 공식: Root * 96 + Quality * 12 + Bass
    """
    bass_idx = state_idx % 12
    qual_idx = (state_idx // 12) % 8
    root_idx = state_idx // 96

    root_str = IDX_TO_PITCH_CLASS[root_idx]
    qual_str = IDX_TO_QUALITY[qual_idx]

    # 루트와 베이스가 같으면 기본 화음, 다르면 슬래시 코드 표기
    if root_idx == bass_idx:
        return f"{root_str}:{qual_str}"
    else:
        bass_str = IDX_TO_PITCH_CLASS[bass_idx]
        return f"{root_str}:{qual_str}/{bass_str}"


def export_debug_pivot_table(
    count_matrix: np.ndarray, filename: str = "chord_transition_debug.csv"
):
    """
    라플라스 스무딩이 적용되기 '전'의 순수 count_matrix를 받아,
    활성화된 화음만 필터링한 후 확률(%)이 명시된 피벗 테이블 CSV를 생성합니다.
    """
    num_states = count_matrix.shape[0]

    # 1. 전체 1152개 화음 이름 리스트 생성
    chord_names = [decode_state_to_chord(i) for i in range(num_states)]

    # 2. Pandas DataFrame 씌우기 (행: 현재 코드, 열: 다음 코드)
    df_counts = pd.DataFrame(count_matrix, index=chord_names, columns=chord_names)

    # 3. '유령 코드' 솎아내기 (가독성 극대화)
    # 데이터셋에 한 번이라도 등장한 적 있는(행 합계나 열 합계가 0보다 큰) 코드만 추려냄
    active_mask = (df_counts.sum(axis=1) > 0) | (df_counts.sum(axis=0) > 0)
    df_active_counts = df_counts.loc[active_mask, active_mask]

    # 4. 인간 친화적인 조건부 확률(%)로 변환
    row_sums = df_active_counts.sum(axis=1)
    # 분모가 0인 경우(전이된 적이 없는 마지막 코드 등) NaN 처리하여 에러 방어
    df_prob = df_active_counts.div(row_sums.replace(0, np.nan), axis=0) * 100

    # 5. 보기 편하게 소수점 둘째 자리까지 반올림 후 저장
    df_prob = df_prob.round(2)

    # NaN은 보기 편하게 0.0이나 빈칸으로 처리 가능 (여기서는 0으로 채움)
    df_prob = df_prob.fillna(0.0)

    df_prob.to_csv(filename)

    # 디버그 요약 정보 출력
    print(
        f"[Debug] 1152개 경우의 수 중 {active_mask.sum()}개 화음만 데이터셋에 존재합니다."
    )
    print(f"[Debug] '{filename}' 파일이 성공적으로 저장되었습니다.")

    return df_prob


if __name__ == "__main__":
    sequences = extract_sequences_from_choco(
        "/Users/seungyeon/workspace/chord_recog/data/choco/jams"
    )
    prob_matrix = compute_chord_transition_matrix_numpy(sequences)
    np.save(
        "/Users/seungyeon/workspace/chord_recog/code/scripts/chord/transition_matrix.npy",
        prob_matrix,
        allow_pickle=True,
    )
    # count_matrix = compute_chord_transition_matrix_numpy(sequences, return_count=True)
    # export_debug_pivot_table(count_matrix)
