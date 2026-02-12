import os
import zipfile
from threading import Thread
from kivy.clock import Clock
from kivy.utils import platform

VOSK_MODEL_URL = "https://alphacephei.com/kaldi/models/vosk-model-small-en-us-0.15.zip"
VOSK_MODEL_DIRNAME = "vosk-model-small-en-us-0.15"

if platform == "android":
    from jnius import autoclass, PythonJavaClass, java_method


class _RecognitionListener(PythonJavaClass):
    __javainterfaces__ = ['org/vosk/android/RecognitionListener']
    __javacontext__ = 'app'

    def __init__(self, on_text, on_final, on_error, on_timeout):
        super().__init__()
        self._on_text = on_text
        self._on_final = on_final
        self._on_error = on_error
        self._on_timeout = on_timeout

    @java_method('(Ljava/lang/String;)V')
    def onPartialResult(self, hypothesis):
        Clock.schedule_once(lambda dt: self._on_text(str(hypothesis)))

    @java_method('(Ljava/lang/String;)V')
    def onResult(self, hypothesis):
        Clock.schedule_once(lambda dt: self._on_text(str(hypothesis)))

    @java_method('(Ljava/lang/String;)V')
    def onFinalResult(self, hypothesis):
        Clock.schedule_once(lambda dt: self._on_final(str(hypothesis)))

    @java_method('(Ljava/lang/Exception;)V')
    def onError(self, exception):
        Clock.schedule_once(lambda dt: self._on_error(str(exception)))

    @java_method('()V')
    def onTimeout(self):
        Clock.schedule_once(lambda dt: self._on_timeout())


class VoskAndroidSTT:
    """
    Offline STT using Vosk Android (AAR) via pyjnius.
    Downloads the small model once into config_dir.
    """

    def __init__(self, config_dir: str, show_toast=None):
        self.config_dir = config_dir
        self.show_toast = show_toast or (lambda *a, **k: None)

        self.model_path = os.path.join(self.config_dir, VOSK_MODEL_DIRNAME)
        self._speech_service = None
        self._model = None

    def ensure_model_async(self, done_cb):
        """
        done_cb(ok: bool, info: str)
        """
        def job():
            try:
                if os.path.isdir(self.model_path) and os.listdir(self.model_path):
                    Clock.schedule_once(lambda dt: done_cb(True, self.model_path))
                    return

                os.makedirs(self.config_dir, exist_ok=True)
                zip_path = os.path.join(self.config_dir, VOSK_MODEL_DIRNAME + ".zip")

                import requests
                self.show_toast("Downloading Vosk model (~40MB)...")
                with requests.get(VOSK_MODEL_URL, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 64):
                            if chunk:
                                f.write(chunk)

                self.show_toast("Extracting Vosk model...")
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(self.config_dir)
                try:
                    os.remove(zip_path)
                except Exception:
                    pass

                ok = os.path.isdir(self.model_path)
                Clock.schedule_once(lambda dt: done_cb(ok, self.model_path if ok else "Model folder not found"))
            except Exception as e:
                Clock.schedule_once(lambda dt: done_cb(False, str(e)))

        Thread(target=job, daemon=True).start()

    def start(self, on_text, on_final, on_error, on_timeout, sample_rate=16000.0):
        if platform != "android":
            on_error("VoskAndroidSTT works only on Android in this setup.")
            return

        try:
            Model = autoclass('org.vosk.Model')
            Recognizer = autoclass('org.vosk.Recognizer')
            SpeechService = autoclass('org.vosk.android.SpeechService')

            if self._model is None:
                self._model = Model(self.model_path)

            recognizer = Recognizer(self._model, float(sample_rate))
            listener = _RecognitionListener(on_text, on_final, on_error, on_timeout)

            self._speech_service = SpeechService(recognizer, float(sample_rate))
            self._speech_service.startListening(listener)

        except Exception as e:
            on_error(str(e))

    def stop(self):
        try:
            if self._speech_service:
                self._speech_service.stop()
                self._speech_service.shutdown()
        except Exception:
            pass
        self._speech_service = None
