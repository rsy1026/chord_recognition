from music21.harmony import CHORD_TYPES

DEFAULT_QPM = 120
RESOLUTION_PER_QUARTER = 48
RESOLUTION_PER_MEASURE = RESOLUTION_PER_QUARTER * 4

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
PITCH_CLASS_TO_IDX = {
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
INTERVAL_TO_IDX = {
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
QUALITY_INTERVALS = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "major-seventh": [0, 4, 7, 11],
    "minor-seventh": [0, 3, 7, 10],
    "dominant-seventh": [0, 4, 7, 10],
    "diminished": [0, 3, 6],
    "half-diminished": [0, 3, 6, 10],
    "augmented": [0, 4, 8],
}
CHORD_DEGREE_TO_KIND = {v[0]: k for k, v in CHORD_TYPES.items()}
CHORD_KINDSTR_TO_KIND = {vv: k for k, v in CHORD_TYPES.items() for vv in v[1]}
REST = 128
