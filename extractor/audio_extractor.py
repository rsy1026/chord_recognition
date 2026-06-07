import os
import time
import yt_dlp
from yt_dlp import utils


class AudioExtractor:

    def __init__(self):
        self.downloaded_files = []

        # 1. 다운로드 완료 시점 파싱 (가장 직관적이고 안전)
        def my_progress_hook(d):
            if d["status"] == "finished":
                base_path = d["filename"]
                # FFmpeg 변환 후 최종적으로 생성될 .wav 경로를 예측하여 등록
                name, _ = os.path.splitext(base_path)
                predicted_wav = f"{name}.wav"
                if predicted_wav not in self.downloaded_files:
                    self.downloaded_files.append(predicted_wav)
                    print(f"\n[Hook] 파일 생성 감지: {predicted_wav}")

        self.ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "%(title)s.%(ext)s",
            "progress_hooks": [my_progress_hook],
            "noplaylist": True,  # 플레이리스트 세션 완전 차단
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            # ---------------------------------------------------------------
            # ★ 무한 루프 절대 방지용 원자폭탄 옵션들 ★
            # ---------------------------------------------------------------
            "force_overwrites": True,  # .part 찌꺼기가 있으면 이어서 받지 말고 밀어버림
            "ignoreerrors": False,  # 조그만 에러라도 나면 즉시 스크립트 중단
            "retries": 0,  # 네트워크 재시도 횟수 0회 강제
            "fragment_retries": 0,  # 대시 프래그먼트 재시도 0회 강제
            "file_access_retries": 0,  # 파일 접근 재시도 0회 강제
            # 유튜브 응답이 3초 넘게 안 오면 서버가 먹통이거나 차단한 것이므로 즉시 에러 발생
            "socket_timeout": 3,
            "timeout": 3,
            # ---------------------------------------------------------------
            # 봇 탐지 우회 헤더
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            },
        }

    def execute(self, url: str):
        start_time = time.time()
        self.downloaded_files = []

        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:  # type: ignore
                ydl.cache.remove()  # 416 에러 유발하는 세션 캐시 제거
                ydl.download([url])

            print(f"최종 완료 파일 리스트: {self.downloaded_files}")
            print(f"소요 시간: {time.time() - start_time:.2f}초")
            return self.downloaded_files[0]

        except utils.DownloadError as e:
            # 루프를 도는 대신 이쪽 예외 영역으로 튕겨져 나옵니다.
            print(f"\n[루프 차단 완료] 유튜브 다운로드 거부 또는 차단 발생: {e}")
            return None

        except Exception as e:
            print(f"기타 예외 발생: {e}")
            return None


if __name__ == "__main__":
    extractor = AudioExtractor()
    target_url = "https://youtu.be/xKT92YmzhJw"
    extracted_filenames = extractor.execute(target_url)
