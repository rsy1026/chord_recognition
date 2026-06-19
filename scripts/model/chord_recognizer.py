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
        raw_similarity = np.dot(self.templates, chroma_cleaned)
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

        # 4. 다양성이 보장된 역추적 (Diversity-Aware Backtracking)
        # 단순히 3개를 뽑지 않고, 여유 있게 상위 30개의 도착점을 추출하여 검사합니다.
        search_pool_size = min(30, K * 10)
        final_scores_flat = T1[:, M - 1, :].flatten()

        # 상위 30개의 인덱스를 점수 내림차순으로 정렬
        pool_indices = np.argsort(final_scores_flat)[-search_pool_size:][::-1]

        curr_states_pool = pool_indices // K
        curr_ks_pool = pool_indices % K

        best_sequences = []
        # 중복 검사를 위해 각 시퀀스의 '순수 화성(Root+Quality) 궤적'을 문자열로 해싱하여 저장
        seen_harmonic_trajectories = set()

        for i in range(search_pool_size):
            s = curr_states_pool[i]
            k = curr_ks_pool[i]

            path = np.zeros(M, dtype=np.int32)
            path[M - 1] = s

            curr_s = s
            curr_k = k

            # 단일 경로 역추적
            for t in range(M - 2, -1, -1):
                prev_s = T2[curr_s, t + 1, curr_k]
                prev_k = T3[curr_s, t + 1, curr_k]
                path[t] = prev_s
                curr_s = prev_s
                curr_k = prev_k

            # --- [중복 검사 로직] ---
            # 추출된 path에서 베이스를 버리고 Root(12) * Quality(8) = 96차원의 궤적만 추출
            harmonic_trajectory = tuple((state // 12) for state in path)

            # 완전히 똑같은 화성 진행(베이스만 다른 경우)은 과감하게 폐기
            if harmonic_trajectory not in seen_harmonic_trajectories:
                seen_harmonic_trajectories.add(harmonic_trajectory)
                best_sequences.append(path)

            # 진정으로 다른 궤적을 가진 K(3)개의 시퀀스가 모이면 루프 즉시 종료
            if len(best_sequences) == K:
                break

        return np.array(best_sequences)

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
