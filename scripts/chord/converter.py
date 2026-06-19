import json
import re
import glob
import numpy as np
import pandas as pd
from typing import Optional

from chord.constants import (
    IDX_TO_PITCH_CLASS,
    QUALITY_TO_IDX,
    IDX_TO_QUALITY,
    PITCH_CLASS_TO_IDX,
    INTERVAL_TO_IDX,
)


class ChordConverter:
    def __init__(self):
        pass

    def save_transition_matrix(self, jams_path: str):
        sequences = self.extract_sequences_from_choco(jams_path)
        relative_tensor = self.build_relative_transition_tensor(sequences)
        log_A = self.project_relative_to_absolute_A(relative_tensor)

        return log_A

    def extract_sequences_from_choco(self, jams_folder_path: str) -> list:
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
                    state_idx = self.parse_choco_harte_label(harte_label)
                    seq.append(state_idx)

                sequences.append(seq)

            except Exception as e:
                # 깨진 JAMS 파일은 과감히 패스
                continue

        return sequences

    @staticmethod
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

        root_idx = PITCH_CLASS_TO_IDX.get(root_str, 0)

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
        if bass_str and bass_str in INTERVAL_TO_IDX:
            bass_interval = INTERVAL_TO_IDX[bass_str]
            bass_idx = (root_idx + bass_interval) % 12
        else:
            bass_idx = root_idx

        # 인덱스 계산 공식도 8 Quality(96개 뼈대) 체제에 맞게 변경
        # Root(12) * 96 + Quality(8) * 12 + Bass(12)
        state_index = root_idx * 96 + qual_idx * 12 + bass_idx

        return state_index

    def build_relative_transition_tensor(self, chord_seqs, num_qualities=8, alpha=0.1):
        """
        절대 조성을 무시하고, 화음의 '퀄리티 전환'과 '근음의 도약 거리'만 통계 처리합니다.
        차원: (8, 12, 8, 12) -> (이전 퀄리티, 루트 도약 거리, 현재 퀄리티, 현재 베이스 전위 상태)
        """
        # 카운트 텐서 초기화
        relative_counts = np.zeros(
            (num_qualities, 12, num_qualities, 12), dtype=np.float64
        )

        for seq in chord_seqs:
            for t in range(len(seq) - 1):
                state_idx_prev = seq[t]
                state_idx_curr = seq[t + 1]

                if state_idx_prev is None or state_idx_curr is None:
                    continue

                # 1152차원 디코딩
                # 이전 화음
                bass_prev = state_idx_prev % 12
                qual_prev = (state_idx_prev // 12) % 8
                root_prev = state_idx_prev // 96

                # 현재 화음
                bass_curr = state_idx_curr % 12
                qual_curr = (state_idx_curr // 12) % 8
                root_idx_curr = state_idx_curr // 96

                # [핵심] 상대적 음정 도약 거리 계산 (Modular 12 아리메틱)
                delta_root = (root_idx_curr - root_prev) % 12
                # 현재 베이스가 현재 루트로부터 얼마나 떨어져 있는지 (전위 형태 기록)
                curr_inversion = (bass_curr - root_idx_curr) % 12

                # 텐서에 카운트 누적
                relative_counts[qual_prev, delta_root, qual_curr, curr_inversion] += 1

        # 라플라스 스무딩 및 정규화
        relative_counts += alpha

        # 이전 상태(qual_prev) 기점으로 모든 가능성의 합이 1.0이 되도록 차원 축 정규화
        # 각 이전 퀄리티 행에서 발생하는 모든 도약의 합으로 나눔
        row_sums = relative_counts.sum(axis=(1, 2, 3), keepdims=True)
        relative_tensor = relative_counts / row_sums

        return np.log(relative_tensor)

    @staticmethod
    def project_relative_to_absolute_A(relative_tensor):
        """
        곡의 조성을 모르는 상태에서, 상대 전이 지식을 절대 1152x1152 행렬로 확장합니다.
        """
        log_A = np.full((1152, 1152), -np.inf)

        for i in range(1152):
            qual_i = (i // 12) % 8
            root_i = i // 96

            for j in range(1152):
                bass_j = j % 12
                qual_j = (j // 12) % 8
                root_j = j // 96

                # 두 절대 상태 간의 물리적 거리 역추적
                delta_root = (root_j - root_i) % 12
                inversion_j = (bass_j - root_j) % 12

                # 텐서 값 복사
                log_A[i, j] = relative_tensor[qual_i, delta_root, qual_j, inversion_j]

        return log_A

    @staticmethod
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
        self, count_matrix: np.ndarray, filename: str = "chord_transition_debug.csv"
    ):
        """
        라플라스 스무딩이 적용되기 '전'의 순수 count_matrix를 받아,
        활성화된 화음만 필터링한 후 확률(%)이 명시된 피벗 테이블 CSV를 생성합니다.
        """
        num_states = count_matrix.shape[0]

        # 1. 전체 1152개 화음 이름 리스트 생성
        chord_names = [self.decode_state_to_chord(i) for i in range(num_states)]

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
    converter = ChordConverter()
    prob_matrix = converter.save_transition_matrix(
        "/Users/seungyeon/workspace/chord_recog/data/choco/jams"
    )
    np.save(
        "/Users/seungyeon/workspace/chord_recog/code/scripts/chord/transition_matrix.npy",
        prob_matrix,
        allow_pickle=True,
    )
    # count_matrix = compute_chord_transition_matrix_numpy(sequences, return_count=True)
    # export_debug_pivot_table(count_matrix)
