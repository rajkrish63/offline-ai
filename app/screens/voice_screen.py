from kivymd.uix.screen import MDScreen
from kivy.properties import StringProperty, BooleanProperty
from kivy.utils import platform
import json


class VoiceScreen(MDScreen):
    transcript = StringProperty("")
    listening = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stt = None

    def _parse_json_text(self, hypothesis_json: str) -> str:
        # hypothesis_json like {"partial":"hi"} or {"text":"hello world"}
        try:
            obj = json.loads(hypothesis_json)
            return obj.get("text") or obj.get("partial") or ""
        except Exception:
            return ""

    def toggle_listen(self):
        from kivymd.app import MDApp
        app = MDApp.get_running_app()

        if platform != "android":
            self.transcript = "Voice STT: Android-only in this setup."
            return

        if self._stt is None:
            from modules.stt_vosk_android import VoskAndroidSTT
            self._stt = VoskAndroidSTT(app.config_dir, show_toast=app.show_toast_msg)

        if self.listening:
            self._stt.stop()
            self.listening = False
            return

        def on_ready(ok, info):
            if not ok:
                self.transcript = f"Model error: {info}"
                return

            self.transcript = "Listening…"
            self.listening = True

            def on_text(h):
                t = self._parse_json_text(h)
                if t:
                    self.transcript = t

            def on_final(h):
                t = self._parse_json_text(h)
                if t:
                    self.transcript = t
                self.listening = False
                self._stt.stop()

            def on_error(e):
                self.transcript = f"STT error: {e}"
                self.listening = False

            def on_timeout():
                self.transcript = "Timeout."
                self.listening = False

            self._stt.start(on_text, on_final, on_error, on_timeout)

        # Ensure model exists first (downloads once)
        self._stt.ensure_model_async(on_ready)

    def send_to_chat(self):
        from kivymd.app import MDApp
        app = MDApp.get_running_app()
        if self.transcript.strip():
            app.inject_text_to_chat(self.transcript.strip(), auto_send=False)
