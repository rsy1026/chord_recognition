import librosa
import numpy as np
import torch
import torchcrepe

from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.stats import mode

from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

from chord.constants import IDX_TO_PITCH_CLASS
from chord.hpcp import generate_hpcp_template_matrix


class FeatureExtractor:
    def __init__(self, sr: int = 22050, hop_len: int = 512):
        self.sr = sr
        self.hop_len = hop_len
        self.template, self.template_key = generate_hpcp_template_matrix()

    def execute(self, audio_path: str | Path):
        print("📊 오디오 특징(Chroma & Beat) 분석 중...")
        y, _ = librosa.load(audio_path, sr=self.sr)

        beats = self.extract_beat(audio_path, sample_size=len(y))
        y_harmonic, _ = librosa.effects.hpss(y, margin=2.0)
        bass_pitch, grid_times = self.extract_bass(y_harmonic, beats)
        chroma = self.extract_chroma(y_harmonic, grid_times)

        return beats, bass_pitch, chroma

    def extract_beat(self, audio_path: str | Path, sample_size: int) -> list:
        """
        2. 오디오를 로드하고 비트 특징 추출 (시간 초 단위)
        """
        # 비트 트래킹 (Non-DL 베이스라인)
        tracker = BeatTrackerPipeline(num_threads=8)
        beats_t = tracker.process_audio(audio_path)
        end_time = sample_size * (1 / self.sr)
        beats_t = beats_t.tolist() + [end_time]

        return beats_t

    def extract_bass(self, y_harmonic: np.ndarray, beats: list) -> tuple:
        y_filtered = self.apply_lowpass_filter(y_harmonic, sr=self.sr)
        bass_pitch, grid_times = self.extract_beat_synchronous_bass(
            y_filtered, beats, sr=self.sr, hop_length=self.hop_len
        )

        return bass_pitch, grid_times

    def extract_chroma(self, y_harmonic: np.ndarray, grid_times: list) -> np.ndarray:
        """
        2. 오디오를 로드하고 크로마(Chroma) 특징 추출
        """
        chroma = self.extract_grid_synchronous_chroma(
            y_harmonic, grid_times, sr=self.sr, hop_length=self.hop_len
        )

        return chroma

    def get_beat_in_frame(self, beats: list) -> list[int]:
        beats_t = [int(b * self.sr // self.hop_len) for b in beats]

        return beats_t

    @staticmethod
    def apply_lowpass_filter(
        y: np.ndarray, sr: float, cutoff_freq: float = 120.0, order: int = 4
    ) -> np.ndarray:
        """
        scipy를 활용한 Zero-phase Butterworth Low-Pass Filter
        """
        # Nyquist 주파수 (샘플링 레이트의 절반)
        nyquist = 0.5 * sr

        # 정규화된 컷오프 주파수 (0 ~ 1 사이 값)
        normal_cutoff = cutoff_freq / nyquist

        # 버터워스 필터 계수 생성
        b, a = butter(order, normal_cutoff, btype="low", analog=False)  # type: ignore

        # filtfilt를 사용해 위상 지연(Phase shift) 없이 필터링 적용
        y_filtered = filtfilt(b, a, y)

        return y_filtered

    @staticmethod
    def extract_beat_synchronous_bass(
        y_bass_only, beat_times, sr, hop_length=512, subdivision=2
    ):
        """
        1. 비트를 16분음표(subdivision=4) 단위의 그리드로 쪼갭니다.
        2. PyIN으로 추출한 프레임 단위의 베이스 피치를 이 그리드에 맞춰
        중앙값(Median)으로 풀링하여 시간축을 완벽하게 정렬합니다.
        """

        # 비트(4분음표) 사이를 선형 보간(Linear Interpolation)하여 세분화
        grid_times = []
        for i in range(len(beat_times) - 1):
            sub_beats = np.linspace(
                beat_times[i], beat_times[i + 1], subdivision, endpoint=False
            )
            grid_times.extend(sub_beats)

        grid_times.append(beat_times[-1])
        grid_times = np.array(grid_times)

        fmin = float(librosa.note_to_hz("C1"))
        fmax = float(librosa.note_to_hz("B3"))

        # --------------------------------------------------------
        # 3. PyIN 베이스 피치 추출 (순수 CPU 구동)
        # --------------------------------------------------------
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y_bass_only,
            fmin=fmin,  # 약 41.2Hz
            fmax=fmax,  # 베이스 대역 상한
            sr=sr,
            frame_length=2048,
            hop_length=hop_length,
            fill_na=np.nan,
        )

        rms = librosa.feature.rms(y=y_bass_only, hop_length=hop_length)[0]

        # 확률 0.2 이상만 유효한 베이스 피치로 인정 (Confidence Filtering)
        valid_idx = voiced_prob > 0.01
        f0_clean = np.full_like(f0, np.nan)
        f0_clean[valid_idx] = f0[valid_idx]

        # Hz를 MIDI 노트 넘버로 변환
        midi_pitch = np.full_like(f0, np.nan)
        midi_pitch[valid_idx] = librosa.hz_to_midi(f0_clean[valid_idx])

        frame_times = librosa.frames_to_time(
            np.arange(len(f0)), sr=sr, hop_length=hop_length
        )

        # --------------------------------------------------------
        # 4. 그리드 풀링 (Median Pooling)
        # --------------------------------------------------------
        pooled_midi = []
        onset_ranges = []

        for i in range(len(grid_times) - 1):
            start_t = grid_times[i]
            end_t = grid_times[i + 1]

            # 16분음표 '방(Grid)' 하나에 해당하는 프레임들만 마스킹
            mask = (frame_times >= start_t) & (frame_times < end_t)
            segment_rms = rms[mask]

            if np.mean(segment_rms) < np.mean(rms) * 0.1:
                continue

            segment_pitches = midi_pitch[mask]

            # NaN을 제외한 유효 피치 궤적만 추출
            valid_segment_pitches = segment_pitches[~np.isnan(segment_pitches)]

            if len(valid_segment_pitches) > 0:
                # 방 안에 유효한 피치가 존재하면 통계적 대푯값(Median) 추출
                median_pitch = mode(valid_segment_pitches).mode
                # 베이스 라인 출력을 위해 반올림하여 정수형 미디 노트로 확정
                pooled_midi.append(np.round(median_pitch))
                onset_ranges.append((start_t, end_t))
            else:
                # 묵음 구간
                # pooled_midi.append(np.nan)
                continue

        return np.array(pooled_midi), onset_ranges

    @staticmethod
    def extract_grid_synchronous_chroma(y_harmonic, grid_times, sr, hop_length=512):
        """
        고해상도 프레임 단위로 추출된 크로마그램을
        주어진 비트 그리드(grid_times)에 맞춰 중앙값(Median)으로 풀링합니다.

        Args:
            y_harmonic: HPSS를 거친 하모닉 오디오 파형
            sr: 샘플링 레이트
            grid_times: 이전 단계에서 구한 16분음표(혹은 8분음표) 단위의 그리드 시간 배열
            hop_length: 원시 크로마 추출용 프레임 간격

        Returns:
            chroma_pooled: (12, M) 차원의 동기화된 크로마그램 행렬
        """
        # --------------------------------------------------------
        # 1. 고해상도 크로마그램 추출
        # --------------------------------------------------------
        # 일반 STFT 기반 크로마보다 음악적 피치 해상도가 높은 CQT(Constant-Q Transform) 기반 사용
        chroma_raw = librosa.feature.chroma_cqt(
            y=y_harmonic, sr=sr, hop_length=hop_length
        )

        # 각 프레임이 나타내는 실제 시간(초)을 계산
        frame_times = librosa.frames_to_time(
            np.arange(chroma_raw.shape[1]), sr=sr, hop_length=hop_length
        )

        # --------------------------------------------------------
        # 2. 풀링(Pooling) 행렬 초기화
        # --------------------------------------------------------
        num_grids = len(grid_times)
        chroma_pooled = np.zeros((12, num_grids))

        # --------------------------------------------------------
        # 3. 비트 그리드 단위 시간축 정렬 및 Median Pooling
        # --------------------------------------------------------
        for i in range(num_grids):
            start_t, end_t = grid_times[i]

            # 현재 그리드 '방(Grid)' 내부에 들어오는 프레임들을 마스킹
            mask = (frame_times >= start_t) & (frame_times < end_t)

            if np.any(mask):
                # 방 안에 있는 크로마 프레임들을 시간축(axis=1) 기준으로 중앙값 연산
                # 노이즈성으로 잠깐 튀는 비화성음을 통계적으로 깎아내는 역할
                chroma_pooled[:, i] = np.median(chroma_raw[:, mask], axis=1)
            else:
                # 프레임이 하나도 배정되지 않는 극단적인 짧은 그리드의 경우 안전장치
                chroma_pooled[:, i] = 0.0

        # --------------------------------------------------------
        # 4. 행렬 정규화 (Normalization)
        # --------------------------------------------------------
        # 각 그리드별 에너지 편차를 줄이기 위해 열(Column) 단위 특성 스케일링
        # 이후 HMM 디코딩 엔진(Gaussian/Multinomial)에 넣을 때 발산(Explosion) 방지
        chroma_pooled = librosa.util.normalize(chroma_pooled, norm=2, axis=0)

        return chroma_pooled


class BeatTrackerPipeline:
    def __init__(self, num_threads=8):
        """
        서버 기동 시 메모리에 단 한 번만 앙상블 모델을 적재합니다.
        num_threads를 할당하여 8개의 BLSTM 모델을 병렬 연산합니다.
        """
        # 관측 확률(Observation) 추출기
        self.rnn_processor = RNNBeatProcessor(num_threads=num_threads)

        # Viterbi 디코더 (fps는 100으로 고정)
        self.dbn_processor = DBNBeatTrackingProcessor(fps=100)

    def process_audio(self, audio_path):
        """
        추론(Inference) 단계에서는 이미 적재된 객체를 재사용합니다.
        """
        # 1. 딥러닝 기반 관측 확률 행렬 도출
        act = self.rnn_processor(audio_path)

        # 2. HMM/Viterbi 디코딩으로 최적의 타임스탬프 도출
        beats = self.dbn_processor(act)

        return beats
