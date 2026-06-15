from music21.harmony import CHORD_TYPES

ROUND_DECIMAL = 6
MAX_MEASURE_LENGTH = 8
MIN_MEASURE_LENGTH = 4
MIN_NOTE_NUM = 4
MIN_CHORD_NUM = 4
OVERLAP_MEASURE_LENGTH = 2
EVAL_MEASURE_LENGTH = MAX_MEASURE_LENGTH
EVAL_PROMPT_MEASURE_LENGTH = 2
DEFAULT_QPM = 120
MAX_SEQ_LEN = 2048

QUARTER_NOTE = 4
EIGHTH_NOTE = 8
SIXTEENTH_NOTE = 16
RESOLUTION_PER_QUARTER = 48
RESOLUTION_PER_MEASURE = RESOLUTION_PER_QUARTER * 4

CHORD_ROOT_RATIO = 1

DIATONIC_TO_IDX = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
ACCIDENTAL_TO_IDX = {"#": 1, "b": -1, "-": -1, "x": 2}
IDX_TO_PITCH_CLASS = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}
QUALITY_TO_IDX = {
    q: n
    for n, q in enumerate(
        [
            "major",
            "minor",
            "augmented",
            "diminished",
            "half-diminished",
            "major-seventh",
            "minor-seventh",
            "dominant-seventh",
        ]
    )
}
IDX_TO_QUALITY = {v: k for k, v in QUALITY_TO_IDX.items()}
PITCH = list(range(129))  # 0 ~ 128 (including REST)
OCTAVE = list(range(-1, 10))  # -1 ~ 9
DURATION_SIMPLE_STR = {
    1: "16th",
    2: "8th",
    3: "dotted_8th",
    4: "quarter",
    5: "1.25_quarter",
    6: "dotted_quarter",
    7: "double_dotted_quarter",
    8: "half",
    10: "1_25_half",
    12: "dotted_half",
    14: "double_dotted_half",
    16: "whole",
    24: "dotted_whole",
}
DURATION_EXTENDED_STR = {
    k + 1: f"{k+1}/{RESOLUTION_PER_QUARTER}_quarter"
    for k in range(RESOLUTION_PER_QUARTER * 6)
}
for k, v in DURATION_SIMPLE_STR.items():
    DURATION_EXTENDED_STR[k] = v
DURATION_STR = DURATION_EXTENDED_STR
DURATION = list(DURATION_STR.keys())
TIMESIG = [
    "4/4",
    "3/4",
    "2/2",
    "6/8",
    "2/4",
    "12/8",
    "9/8",
    "6/4",
    "3/2",
]
CHORD_KIND = sorted(CHORD_TYPES.keys())
OUT_OF_CHORD_TYPES_MAP = {
    "minor-major": "major",
    "augmented-ninth": "augmented-dominant-ninth",
    "maj69": "major",
    "5": "power",
}
TENSION_TONE = ["9", "11", "13"]
TENSION = [acc + t for t in TENSION_TONE for acc in ["b", "#", "x"]]
CHORD_NAME = [cr + ck for cr in IDX_TO_PITCH_CLASS.values() for ck in CHORD_KIND]
CHORD_DEGREE_TO_KIND = {v[0]: k for k, v in CHORD_TYPES.items()}
CHORD_KINDSTR_TO_KIND = {vv: k for k, v in CHORD_TYPES.items() for vv in v[1]}
INVALID_CHORD_KINDS = {"", "none"}
CHORD_TONE_SIMPLE_MAP = {
    "m": {0, 3, 7},
    "6": {0, 4, 7, 9},
    "maj7": {0, 4, 7, 11},
    "7": {0, 4, 7, 10},
    "dim": {0, 3, 6},
    "m6": {0, 3, 7, 9},
    "aug7": {0, 4, 8, 10},
    "m7": {0, 3, 7, 10},
    "": {0, 4, 7},
    "9": {0, 4, 10, 2},  # (1,3,b7,9)
    "aug": {0, 4, 8},
    "13": {0, 4, 10, 9},  # (1,3,b7,13)
    "dim7": {0, 3, 6, 9},
    "m7b5": {0, 3, 6, 10},
    "sus": {0, 5, 7},
    "5": {0, 7},
    "m11": {0, 3, 10, 5},  # (1,b3,b7,11)
    "aug9": {0, 4, 10, 3},  # (1,3,b7,#9)
    "m9": {0, 3, 10, 2},  # (1,b3,b7,9)
    "maj9": {0, 4, 11, 2},  # (1,3,7,9)
    "m(maj7)": {0, 3, 7, 11},  # (1,b3,5,7)
    "11": {0, 10, 2, 5},  # (1,b7,9,11)
    "6(add9)": {0, 4, 9, 2},  # (1,3,6,9)
    "sus2": {0, 2, 7},
    "sus7": {0, 5, 7, 10},
    "m13": {0, 3, 10, 9},  # (1,b3,b7,13)
    "maj13": {0, 4, 11, 9},  # (1,3,7,13)
}
CHORD_TONE_FULL_MAP = {
    "m": {0, 3, 7},
    "6": {0, 4, 7, 9},
    "maj7": {0, 4, 7, 11},
    "7": {0, 4, 7, 10},
    "dim": {0, 3, 6},
    "m6": {0, 3, 7, 9},
    "aug7": {0, 4, 8, 10},
    "m7": {0, 3, 7, 10},
    "": {0, 4, 7},
    "9": {0, 4, 7, 10, 2},  # (1,3,b7,9)
    "aug": {0, 4, 8},
    "13": {0, 4, 7, 10, 2, 5, 9},  # (1,3,b7,13)
    "dim7": {0, 3, 6, 9},
    "m7b5": {0, 3, 6, 10},
    "sus": {0, 5, 7},
    "5": {0, 7},
    "m11": {0, 3, 7, 10, 2, 5},  # (1,b3,b7,11)
    "aug9": {0, 4, 7, 10, 3},  # (1,3,b7,#9)
    "m9": {0, 3, 7, 10, 2},  # (1,b3,b7,9)
    "maj9": {0, 4, 7, 11, 2},  # (1,3,7,9)
    "m(maj7)": {0, 3, 7, 11},  # (1,b3,5,7)
    "11": {0, 4, 7, 10, 2, 5},  # (1,b7,9,11)
    "6(add9)": {0, 4, 7, 9, 2},  # (1,3,6,9)
    "sus2": {0, 2, 7},
    "sus7": {0, 5, 7, 10},
    "m13": {0, 3, 7, 10, 2, 5, 9},  # (1,b3,b7,13)
    "maj13": {0, 4, 7, 11, 2, 5, 9},  # (1,3,7,13)
}

CONSONANCE_INTERVALS = [0, 3, 4, 7, 8, 9]
PITCH_TO_IDX = {k: v for v, k in enumerate(PITCH)}
OCTAVE_TO_IDX = {k: v for v, k in enumerate(OCTAVE)}
DURATION_TO_IDX = {k: v for v, k in enumerate(DURATION)}
CHORD_KIND_TO_IDX = {k: v for v, k in enumerate(CHORD_KIND)}
TIMESIG_TO_IDX = {k: v for v, k in enumerate(TIMESIG)}
CHORD_NAME_TO_IDX = {k: v for v, k in enumerate(CHORD_NAME)}
PITCH_CLASS_TO_IDX = {v: k for k, v in IDX_TO_PITCH_CLASS.items()}
IDX_TO_PITCH = {v: k for k, v in PITCH_TO_IDX.items()}
IDX_TO_OCTAVE = {v: k for k, v in OCTAVE_TO_IDX.items()}
IDX_TO_DURATION = {v: k for k, v in DURATION_TO_IDX.items()}
IDX_TO_DURATION_STR = {DURATION_TO_IDX[k]: v for k, v in DURATION_STR.items()}
IDX_TO_CHORD_KIND = {v: k for k, v in CHORD_KIND_TO_IDX.items()}
IDX_TO_TIMESIG = {v: k for k, v in TIMESIG_TO_IDX.items()}
IDX_TO_CHORD_NAME = {v: k for k, v in CHORD_NAME_TO_IDX.items()}

REST = PITCH[-1]  # 128
SUPPORTED_PATTERNS = {
    "ii-V-I": "2-5-1 진행 (투파이브원, 251 등)",
    "line_cliche": "라인 클리셰 (베이스 하행/상행 등)",
    "modal_interchange": "모달 인터체인지 (차용화음 등)",
    "diminished_passing": "디미니쉬 패싱 코드",
    "pedal_point": "페달 포인트 (베이스 고정)",
}
