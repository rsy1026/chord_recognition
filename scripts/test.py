import time
import librosa
import numpy as np
import yt_dlp
from pathlib import Path

import subprocess


def download_youtube_audio(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "external_downloader": "aria2c",
        "external_downloader_args": ["-x", "16", "-s", "16", "-k", "1M"],
        "outtmpl": "%(title)s.%(ext)s",
        "noplaylist": True,  # 플레이리스트 여부와 상관없이 단일 곡만 보장
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    start_time = time.time()

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
            # 💡 핵심: download=True로 다운로드를 수행하면서 동시에 메타데이터 딕셔너리를 가져옵니다.
            info_dict = ydl.extract_info(url, download=True)

            # 💡 yt-dlp가 포스트 프로세싱(WAV 변환)까지 마친 최종 파일 경로를 역으로 추출합니다.
            final_filename = ydl.prepare_filename(info_dict)

            # 원래 확장자(webm, m4a 등)를 .wav로 수동 치환해 줍니다. (FFmpeg 변환 반영)
            if final_filename.endswith((".webm", ".m4a", ".mp3", ".opus")):
                final_filename = final_filename.rsplit(".", 1)[0] + ".wav"
            elif not final_filename.endswith(".wav"):
                final_filename += ".wav"

        print(f"다운로드 완료: {final_filename}")
        print(f"소요 시간: {time.time() - start_time:.2f}초")

        # 🚀 다음 파이프라인(librosa.load 등)으로 넘겨주기 위해 파일명을 리턴합니다.
        return final_filename

    except Exception as e:
        print(f"오류 발생: {e}")
        return None


def extract_features(audio_path):
    """
    2. 오디오를 로드하고 비트 및 크로마(Chroma) 특징 추출
    """
    print("📊 오디오 특징(Chroma & Beat) 분석 중...")
    y, sr = librosa.load(audio_path, sr=22050)

    # 비트 트래킹 (Non-DL 베이스라인)
    audio_path = Path(audio_path)
    out_path = audio_path.parent / f"{audio_path.stem}.beat.txt"
    subprocess.call(["DBNBeatTracker", "single", "-o", out_path, audio_path])

    with open(out_path, "r") as f:
        beats = list(map(lambda x: float(x.strip()), f.readlines()))

    # 오디오 주파수 특징 추출 (CQT 기반 크로마 벡터)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    return chroma, beats, sr


def simple_chord_mapping(chroma):
    """
    3. [베이스라인용 규칙 기반 알고리즘]
       딥러닝 모델 이식 전, 크로마 벡터와 가장 유사한 메이저/마이너 코드를 매핑
    """
    # 단순화를 위한 12키 메이저 트라이어드 템플릿 정의 (C, C#, D...)
    # 딥러닝 모델을 붙이기 전 파이프라인 입출력이 정상 작동하는지 확인하는 디버깅용입니다.
    major_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])

    chords_result = []
    # 시간축(프레임) 단위로 순회
    for frame in chroma.T:
        # 코사인 유사도 등이 들어갈 자리 (여기선 단순 점곱)
        best_match = "C Major"  # 예시성 타겟
        chords_result.append(best_match)

    return chords_result


if __name__ == "__main__":
    # 테스트할 유튜브 링크 입력
    TEST_URL = "https://youtu.be/xKT92YmzhJw?si=uzUI0hrQkSrP2PSY"

    # 1. 다운로드
    # audio_file = download_youtube_audio(TEST_URL)
    audio_file = "/Users/seungyeon/workspace/chord_recog/code/scripts/Crush (크러쉬) - 'UP ALL NITE (Feat. SUMIN)' MV.wav"

    # 2. 특징 추출
    chroma, beats, sr = extract_features(audio_file)
    print(f"추출 완료 - 크로마 형태: {chroma.shape}, 감지된 비트 수: {len(beats)}")

    # 3. 임시 매핑 및 출력 테스트
    # 이 단계를 거친 후 딥러닝 모델(Autochord나 커스텀 모델)의 Inference 코드를 연결하면 됩니다.
    print(
        "🚀 기본 파이프라인 골격이 성공적으로 작동합니다. 이제 모델을 얹을 차례입니다."
    )

    # 사용이 끝난 임시 파일 정리
    # if os.path.exists(audio_file):
    #     os.remove(audio_file)
