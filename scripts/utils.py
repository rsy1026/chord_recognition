def set_seed(seed):
    import os
    import random
    import numpy as np
    import torch
    from torch.backends import cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)  # 일부 해시 기반 불확정성 제거


import re
import json
import logging
import numpy as np
from pathlib import Path
from decimal import ROUND_HALF_UP, getcontext
from fractions import Fraction

from music21.harmony import ChordSymbol, getAbbreviationListGivenChordType

from chord.constants import (
    RESOLUTION_PER_MEASURE,
    IDX_TO_PITCH_CLASS,
    DIATONIC_TO_IDX,
    ACCIDENTAL_TO_IDX,
    CHORD_KINDSTR_TO_KIND,
    REST,
)

dc = getcontext()
dc.prec = 48
dc.rounding = ROUND_HALF_UP


def idx2str(idx: int, str_len: int = 3):
    str_idx = str(idx)
    return (str_len - len(str_idx)) * "0" + str_idx


def load_json(filename: str | Path) -> dict:
    with open(filename) as f:
        data = json.load(f)
    return data


def get_chord_symbol(root: str | int, kind_str: str) -> ChordSymbol:
    if isinstance(root, int):
        root = IDX_TO_PITCH_CLASS[root]
    kind_str_name = kind_str.split("(")[0]
    if len(kind_str.split("(")) == 2:
        tension = kind_str.split("(")[1].replace(")", "")
    else:
        tension = ""
    name = root + getAbbreviationListGivenChordType(kind_str_name)[0] + tension
    h = ChordSymbol(name)
    return h


def get_chord_tone(root: str | int, kind_str: str, to_str: bool) -> list:
    h = get_chord_symbol(root, kind_str)
    pitch_names = [re.sub(r"\d+", "", str(p)) for p in h.pitches]
    if to_str:
        return pitch_names
    else:
        chord_tone = [pitch_to_pitch_int(p, to_pc=True) for p in pitch_names]
        return chord_tone


def get_chord_tone_vector(root: str | int, kind_str: str) -> list:
    chord_tone_vector = np.zeros([12])
    chord_tone = get_chord_tone(root, kind_str, to_str=False)
    for index in chord_tone:
        chord_tone_vector[index] = 1
    return chord_tone_vector.tolist()


def get_chord_tone_vector_from_list(chord_tone_list: list) -> list:
    pitch_names = [re.sub(r"\d+", "", str(p)) for p in chord_tone_list]
    chord_tone = {pitch_to_pitch_int(p, to_pc=True) for p in pitch_names}
    chord_tone_vector = np.zeros([12])
    for index in chord_tone:
        chord_tone_vector[index] = 1
    return chord_tone_vector.tolist()


def pitch_to_pitch_int(pitch_str: str, to_pc: bool) -> int:
    alphabet = re.findall(r"[A-Z]", pitch_str)  # matches sequences of alphabets
    accidentals = re.findall(r"[#b\-x]", pitch_str)
    octaves = re.findall(r"[0-9]", pitch_str)
    assert len(alphabet) == 1
    pitch_int = DIATONIC_TO_IDX[alphabet[0]]
    accidentals_int = [ACCIDENTAL_TO_IDX[acc] for acc in accidentals]
    if any(octaves):
        octave_int = int(octaves[0])
    else:
        octave_int = -1
        to_pc = True
    pitch_int += sum(accidentals_int)
    if to_pc:
        return pitch_int % 12
    else:
        return pitch_int + 12 * (octave_int + 1)


def pitch_int_to_pitch(pitch_int: int, to_pc: bool) -> str:
    if pitch_int == REST:  # rest
        return "rest"
    octave, pc_int = np.divmod(pitch_int, 12)
    if to_pc or octave == 0:
        return IDX_TO_PITCH_CLASS[pc_int]
    pitch = IDX_TO_PITCH_CLASS[pc_int] + str(octave - 1)
    return pitch


def chord_name_to_kind(name: str) -> str:
    name = name.split("/")[0]  # remove bass
    kind = re.sub(r"[A-Z]|[#b\-x]", "", name)
    if kind in CHORD_KINDSTR_TO_KIND:
        return CHORD_KINDSTR_TO_KIND[kind]
    else:
        return "other"


def get_num_beat_from_time_sig(
    time_sig: str, beat: int = RESOLUTION_PER_MEASURE
) -> int:
    frac = Fraction(time_sig)
    denom = frac.denominator
    num = frac.numerator
    num_beat = beat / denom * num
    assert num_beat % 1 == 0
    return int(num_beat)


def get_chord_attr_list(chords: list[dict], attr_name: str) -> list:
    return [c[attr_name] for c in chords]


def get_root(
    root_str: str, tonic: int, normalize_to_c: bool, to_str: bool
) -> str | int:
    root = pitch_to_pitch_int(root_str, to_pc=True)
    if normalize_to_c:
        root = normalize_pitch(root, tonic, to_pc=True, to_str=to_str)
    else:
        root = IDX_TO_PITCH_CLASS[root] if to_str else root
    return root


def normalize_pitch(
    pitch: str | int, tonic: int, to_pc: bool = False, to_str: bool = False
) -> int | str:
    if isinstance(pitch, str):
        pitch = pitch_to_pitch_int(pitch, to_pc=False)
    if pitch == REST:
        return pitch
    new_pitch = pitch - tonic
    new_pitch = new_pitch % 12 if to_pc else new_pitch
    new_pitch = pitch_int_to_pitch(new_pitch, to_pc=to_pc) if to_str else new_pitch
    return new_pitch


def normalize_pitches(
    pitches: list[str | int], tonic: int, normalize_to_c: bool, to_str: bool
):
    if normalize_to_c:
        new_pitches = [normalize_pitch(m, tonic, to_str=to_str) for m in pitches]
    else:
        new_pitches = pitches
    return new_pitches


def convert_key_sig_to_major(constants: dict, logger: logging.Logger) -> None:
    tonic = constants["key_sig"]["tonic"]
    mode = constants["key_sig"]["mode"]
    old_key = IDX_TO_PITCH_CLASS[tonic] + f" {mode}"
    if mode == "minor":
        new_tonic = (tonic + 3) % 12
        new_mode = "major"
        constants["key_sig"]["tonic"] = new_tonic
        constants["key_sig"]["mode"] = new_mode
        new_key = IDX_TO_PITCH_CLASS[new_tonic] + f" {new_mode}"
        logger.info(f">> UPDATE | key signature changed: {old_key} -> {new_key}")
