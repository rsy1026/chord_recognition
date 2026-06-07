import os
import shutil
from pathlib import Path

import subprocess

from extractor.audio_extractor import AudioExtractor


def extract_chord(audio_path):
    audio_path = Path(audio_path)
    os.chdir("/Users/seungyeon/workspace/chord_recog/code/BTC_model")

    subprocess.call(
        [
            "python",
            "test.py",
            "--audio_dir",
            audio_path.parent,
            "--save_dir",
            "./",
            "--voca",
            "True",
        ]
    )


if __name__ == "__main__":
    # 테스트할 유튜브 링크 입력
    TEST_URL = "https://youtu.be/xKT92YmzhJw"

    # 1. 다운로드
    audio_file = AudioExtractor().execute(TEST_URL)
    # audio_file = "/Users/seungyeon/workspace/chord_recog/code/scripts/Crush (크러쉬) - 'UP ALL NITE (Feat. SUMIN)' MV.wav"

    shutil.copy(
        str(audio_file),
        "/Users/seungyeon/workspace/chord_recog/code/BTC_model/test/example.wav",
    )

    # 2. 특징 추출
    # chroma, beats, sr = extract_features(audio_file)
    # print(f"추출 완료 - 크로마 형태: {chroma.shape}, 감지된 비트 수: {len(beats)}")
    extract_chord(audio_file)
    print(f"추출 완료")

    # 3. 임시 매핑 및 출력 테스트
    # 이 단계를 거친 후 딥러닝 모델(Autochord나 커스텀 모델)의 Inference 코드를 연결하면 됩니다.
    print(
        "🚀 기본 파이프라인 골격이 성공적으로 작동합니다. 이제 모델을 얹을 차례입니다."
    )

    # 사용이 끝난 임시 파일 정리
    # if os.path.exists(audio_file):
    #     os.remove(audio_file)
