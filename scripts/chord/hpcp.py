import numpy as np

from chord.constants import QUALITY_INTERVALS, QUALITY_TO_IDX


def generate_hpcp_chord_templates():
    """
    1152개 상태 각각에 대해 음향학적 배음 구조(Harmonics)를 반영한
    12차원 HPCP 기반 크로마 템플릿을 생성합니다.
    """
    # 8개 퀄리티별 순수 구성음 간격
    base_intervals = {QUALITY_TO_IDX[k]: v for k, v in QUALITY_INTERVALS.items()}

    # 배음 모델링 가중치 (f, 2f, 3f, 4f, 5f)
    # 기본음 정보 외에 5도(7)와 3도(4) 대역으로 에너지가 자연스럽게 흐르도록 유도
    harmonic_weights = {
        0: 1.0,  # 기본음 (Root)
        7: 0.33,  # 3배음 (완전5도 성분)
        4: 0.20,  # 5배음 (장3도 성분)
    }
    minor_idxs = [
        QUALITY_TO_IDX[q]
        for q in ["minor", "diminished", "minor-seventh", "half-diminished"]
    ]

    templates = np.zeros((1152, 12))

    for state_idx in range(1152):
        qual_idx = (state_idx // 12) % 8
        root_idx = state_idx // 96

        # 1. 해당 화음의 순수 구성음들을 먼저 배치
        chord_notes = [
            (root_idx + interval) % 12 for interval in base_intervals[qual_idx]
        ]

        # 2. 각 구성음이 발생시키는 배음 에너지를 템플릿 12개 칸에 누적 주입
        for note in chord_notes:
            for h_interval, weight in harmonic_weights.items():
                h_note = (note + h_interval) % 12
                templates[state_idx, h_note] += weight

        # 2. [핵심 오답 저격] 마이너 계열 화음(min, min7, hdim, dim -> 1, 5, 7, 2)에
        # 메이저 3도(Root + 4)가 들어오면 내적 점수를 깎아버리는 역가중치(Penalty) 주입
        if qual_idx in minor_idxs:  # 마이너 속성들
            major_3rd_note = (root_idx + 4) % 12
            # 마이너 화음인데 오디오에 메이저 3도(E) 에너지가 존재한다면 점수를 강제로 차감
            templates[state_idx, major_3rd_note] -= 0.4

            # 마이너 정체성인 단3도(Root + 3) 에너지 요구치 강화
            minor_3rd_note = (root_idx + 3) % 12
            templates[state_idx, minor_3rd_note] *= 1.3

    # 3. 크로마그램 데이터(L2 Norm 정규화 처리됨)와
    # 내적(Dot Product) 연산 시 정확한 스케일을 맞추기 위해 템플릿도 행 단위 정규화
    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    hpcp_templates = templates / np.where(norms == 0, 1.0, norms)

    return hpcp_templates
