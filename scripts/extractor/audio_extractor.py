import time
import yt_dlp


class AudioExtractor:
    def __init__(self):
        self.ydl_opts = {
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

    def execute(self, url: str):
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


if __name__ == "__main__":
    # 테스트할 유튜브 링크 입력
    TEST_URL = "https://youtu.be/xKT92YmzhJw?si=uzUI0hrQkSrP2PSY"

    # 1. 다운로드
    audio_file = AudioExtractor().execute(TEST_URL)
    print(f"🚀 오디오 추출 완료: {audio_file}")
