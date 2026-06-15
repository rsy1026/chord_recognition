import numpy as np

from chord.constants import IDX_TO_PITCH_CLASS


def generate_hpcp_template_matrix() -> tuple:
    """
    12개의 루트와 7개의 퀄리티, 1개의 No Chord를 포함하는
    85 x 12 크기의 배음 템플릿 행렬을 생성합니다.
    """
    # 7가지 타겟 퀄리티의 반음 간격(Interval) 정의
    quality_intervals = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "major-seventh": [0, 4, 7, 11],
        "minor-seventh": [0, 3, 7, 10],
        "dominant-seventh": [0, 4, 7, 10],
        "diminished": [0, 3, 6],
        "half-diminished": [0, 3, 6, 10],
        "augmented": [0, 4, 8],
    }

    # 자연 배음렬 가중치 (1배음: 1.0, 3배음(+7반음): 0.33, 5배음(+4반음): 0.2)
    # 옥타브 배음은 크로마에서 같은 피치 클래스로 합산되므로 생략하거나 미세 조정 가능
    overtone_weights = {
        0: 1.0,  # 자기 자신 (Fundamental)
        7: 0.33,  # 완전 5도 (Perfect 5th)
        4: 0.20,  # 장 3도 (Major 3rd)
    }

    templates = []
    template_keys = []

    # 1. 12개 루트 x 7개 퀄리티 = 84개 화음 템플릿 생성
    for root in range(12):
        for q_name, intervals in quality_intervals.items():
            chroma_vector = np.zeros(12)

            # 각 코드톤에 대해 배음 가중치 누적
            for interval in intervals:
                chord_tone = (root + interval) % 12

                for overtone_interval, weight in overtone_weights.items():
                    harmonic_pitch = (chord_tone + overtone_interval) % 12
                    chroma_vector[harmonic_pitch] += weight

            # L2 정규화 (코사인 유사도 계산을 위해 벡터 길이를 1로 맞춤)
            norm = np.linalg.norm(chroma_vector)
            if norm > 0:
                chroma_vector = chroma_vector / norm

            templates.append(chroma_vector)
            template_keys.append("_".join([IDX_TO_PITCH_CLASS[root], q_name]))

    # 2. No Chord (NC) 템플릿 추가 (모든 음이 평탄하거나 에너지가 없는 상태)
    nc_vector = np.ones(12) / np.sqrt(12)  # 평탄한 분포의 L2 정규화
    templates.append(nc_vector)
    template_keys.append("NC")

    # 최종 행렬 변환: (85, 12) 차원
    template_matrix = np.array(templates)

    return template_matrix, template_keys
