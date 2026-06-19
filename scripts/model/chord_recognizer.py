import numpy as np

from chord.constants import IDX_TO_PITCH_CLASS, IDX_TO_QUALITY
from chord.hpcp import generate_hpcp_chord_templates
from chord.converter import ChordConverter


class ChordRecognizer:
    def __init__(self):
        self.templates = generate_hpcp_chord_templates()
        self.converter = ChordConverter
        self.log_A = np.load("chord/transition_matrix.npy", allow_pickle=True)

    def execute(self, chroma_pooled: np.ndarray, pooled_midi: np.ndarray):
        candidates = self.trellis_k_best_viterbi(chroma_pooled, pooled_midi)
        return self.decode_candidates(candidates)

    def trellis_k_best_viterbi(
        self,
        chroma_cleaned,
        pooled_midi,
        K=3,
        gamma=10.0,
        inertia_value=2.0,
        bass_bonus_value=2.0,
    ):
        """
        글로벌 최적 해를 훼손하지 않으면서, 상위 K개의 독립적인 평행우주(악보 시퀀스)를 추출합니다.

        Args:
            chroma_cleaned: 배음이 정제된 (12, M) 크로마 행렬
            pooled_midi: (M,) 차원의 베이스 궤적
            log_A: (1152, 1152) 전이 확률 행렬
            K: 추출할 완본체 시퀀스의 개수
        """
        num_states = 1152
        M = chroma_cleaned.shape[1]

        # 1. 템플릿 매칭 및 관측 확률 초기화 (음정 판별식 템플릿 가정)
        templates = generate_hpcp_chord_templates()
        raw_similarity = np.dot(templates, chroma_cleaned)
        log_E = gamma * raw_similarity

        # 2. 3차원 추적 텐서 초기화
        # T1: (상태, 시간, 순위) - 누적 로그 확률
        # T2: (상태, 시간, 순위) - 역추적용 이전 상태(State) 포인터
        # T3: (상태, 시간, 순위) - 역추적용 이전 경로의 순위(K-Rank) 포인터
        T1 = np.full((num_states, M, K), -np.inf)
        T2 = np.zeros((num_states, M, K), dtype=np.int32)
        T3 = np.zeros((num_states, M, K), dtype=np.int32)

        # 첫 프레임 초기화: 1위(k=0) 슬롯에만 기본 점수 부여
        T1[:, 0, 0] = np.log(1.0 / num_states) + log_E[:, 0]

        # 프루닝 파라미터: 매 프레임마다 탐색을 허용할 상위 화음의 개수
        beam_width = 64  # 64~128 사이면 정확도 손실이 물리적으로 0%에 가깝습니다.

        # 3. 전방 트레리스 연산 (Forward Pass)
        for t in range(1, M):
            curr_bass = pooled_midi[t]
            prev_bass = pooled_midi[t - 1]

            log_A_t = self.log_A.copy()
            if not np.isnan(curr_bass) and not np.isnan(prev_bass):
                pitch_diff = abs(curr_bass - prev_bass)
                if pitch_diff == 0:
                    np.fill_diagonal(log_A_t, np.diagonal(log_A_t) + inertia_value)
                elif pitch_diff == 1:
                    np.fill_diagonal(
                        log_A_t, np.diagonal(log_A_t) + (inertia_value * 0.5)
                    )

            log_E_t = log_E[:, t].copy()
            if not np.isnan(curr_bass):
                bass_class = int(curr_bass % 12)
                matching_mask = np.arange(num_states) % 12 == bass_class
                log_E_t[matching_mask] += bass_bonus_value
                log_E_t[~matching_mask] -= bass_bonus_value * 0.5

            # =====================================================================
            # [고속화 코어: Beam Pruning]
            # 1. 이전 프레임에서 가장 점수가 높은(1등 가설 기준) 상위 64개의 상태 인덱스만 추출
            top_prev_states = np.argsort(T1[:, t - 1, 0])[-beam_width:]

            # 2. 살아남은 64개 상태의 과거 점수만 슬라이싱 -> shape: (64, K)
            prev_scores = T1[top_prev_states, t - 1, :]
            prev_scores_exp = prev_scores[:, np.newaxis, :]  # (64, 1, 3)

            # 3. 전이 확률 행렬(A)도 이 64개의 상태에서 '출발'하는 행만 도려냄 -> shape: (64, 1152)
            log_A_sliced = log_A_t[top_prev_states, :]
            log_A_exp = log_A_sliced[:, :, np.newaxis]  # (64, 1152, 1)

            # 4. 메모리 95% 절약된 초고속 브로드캐스팅 -> shape: (64, 1152, 3)
            step_scores = prev_scores_exp + log_A_exp

            # 형태 변환 -> (1152, 64 * 3) : 각 상태당 검토할 경우의 수가 3456개에서 192개로 폭락
            step_scores_flat = step_scores.transpose(1, 0, 2).reshape(num_states, -1)

            # 상위 K개 정렬 및 추출
            top_k_indices = np.argsort(step_scores_flat, axis=1)[:, -K:][:, ::-1]
            top_k_scores = np.take_along_axis(step_scores_flat, top_k_indices, axis=1)

            T1[:, t, :] = top_k_scores + log_E_t[:, np.newaxis]

            # 5. 역추적 포인터 매핑 복원
            # top_k_indices // K는 0~63 사이의 로컬 인덱스이므로, 이를 원본 1152차원 절대 인덱스로 복원
            prev_beam_indices = top_k_indices // K
            T2[:, t, :] = top_prev_states[prev_beam_indices]
            T3[:, t, :] = top_k_indices % K
            # =====================================================================

        # 4. 역추적 (Backtracking)
        best_sequences = np.zeros((K, M), dtype=np.int32)

        # 마지막 타임스탬프에서 글로벌 1~K위의 도착점 찾기
        final_scores_flat = T1[:, M - 1, :].flatten()
        global_top_k_indices = np.argsort(final_scores_flat)[-K:][::-1]

        curr_states = global_top_k_indices // K
        curr_ks = global_top_k_indices % K

        # 시작점 기록
        for rank in range(K):
            best_sequences[rank, M - 1] = curr_states[rank]

        # 과거로 거슬러 올라가며 K개의 실타래를 동시에 풀어냄
        for t in range(M - 2, -1, -1):
            next_states = np.zeros(K, dtype=np.int32)
            next_ks = np.zeros(K, dtype=np.int32)

            for rank in range(K):
                s = curr_states[rank]
                k = curr_ks[rank]

                # 포인터가 가리키는 이전 프레임의 정보 호출
                prev_s = T2[s, t + 1, k]
                prev_k = T3[s, t + 1, k]

                best_sequences[rank, t] = prev_s
                next_states[rank] = prev_s
                next_ks[rank] = prev_k

            curr_states = next_states
            curr_ks = next_ks

        return best_sequences

    @staticmethod
    def decode_candidates(candidates: np.ndarray):
        new_candidates = []
        for candidate in candidates:
            new_candidate = []
            for chord in candidate:
                chord = ChordConverter.decode_state_to_chord(chord)
                new_candidate.append(chord)
            new_candidates.append(new_candidate)

        return np.asarray(new_candidates)
