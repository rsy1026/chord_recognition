import librosa
import subprocess

from pathlib import Path


class FeatureExtractor:
    def __init__(self):
        pass

    def execute(self, audio_path: str | Path):
        print("📊 오디오 특징(Chroma & Beat) 분석 중...")
        beats = self.extract_beat(audio_path)
        chroma, sr = self.extract_chroma(audio_path)
        return beats, chroma, sr

    def extract_beat(self, audio_path: str | Path) -> list:
        """
        2. 오디오를 로드하고 비트 특징 추출
        """
        # 비트 트래킹 (Non-DL 베이스라인)
        audio_path = Path(audio_path)
        out_path = audio_path.parent / f"{audio_path.stem}.beat.txt"
        subprocess.call(["DBNBeatTracker", "single", "-o", out_path, audio_path])

        with open(out_path, "r") as f:
            beats = list(map(lambda x: float(x.strip()), f.readlines()))

        return beats

    def extract_chroma(self, audio_path: str | Path) -> tuple:
        """
        2. 오디오를 로드하고 크로마(Chroma) 특징 추출
        """
        y, sr = librosa.load(audio_path, sr=22050)

        # 오디오 주파수 특징 추출 (CQT 기반 크로마 벡터)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        return chroma, sr


if __name__ == "__main__":
    # 테스트할 음악
    audio_file = "/Users/seungyeon/workspace/chord_recog/code/scripts/Crush (크러쉬) - 'UP ALL NITE (Feat. SUMIN)' MV.wav"

    # 2. 특징 추출
    beats, chroma, sr = FeatureExtractor().execute(audio_file)
    print(f"🚀 추출 완료 - 크로마 형태: {chroma.shape}, 감지된 비트 수: {len(beats)}")
