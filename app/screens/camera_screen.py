from kivymd.uix.screen import MDScreen
from kivy.properties import StringProperty
from kivy.utils import platform


class CameraScreen(MDScreen):
    ocr_text = StringProperty("Pick an image → OCR → send to chat.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._picked_path = None
        self._ocr = None

    def pick_image(self):
        # Reuse existing MDFileManager from main.py
        from kivymd.app import MDApp
        app = MDApp.get_running_app()
        app.open_doc_picker(from_screen="camera")  # uses same picker (jpg/png allowed)

    def process_image_path(self, path: str):
        self._picked_path = path
        if platform != "android":
            self.ocr_text = "OCR: Android ML Kit setup. (Desktop can use Tesseract.)"
            return
        # --- fast preprocessing for better OCR ---
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import os, time

            img = Image.open(path)

            # 1) convert + rotate using EXIF if present
            try:
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass

            # 2) resize (keeps OCR sharp but avoids huge photos)
            max_w = 1600  # good for low-end phones
            if img.width > max_w:
                ratio = max_w / float(img.width)
                img = img.resize((max_w, int(img.height * ratio)))

            # 3) improve contrast + sharpness
            img = img.convert("RGB")
            img = ImageEnhance.Contrast(img).enhance(1.6)
            img = ImageEnhance.Sharpness(img).enhance(2.0)
            img = img.filter(ImageFilter.MedianFilter(size=3))

            # save temp file
            from kivymd.app import MDApp
            app = MDApp.get_running_app()
            tmp_path = os.path.join(app.in_dir, f"ocr_{int(time.time())}.jpg")
            img.save(tmp_path, quality=90)

            path_for_ocr = tmp_path
        except Exception:
            # if preprocessing fails, fallback to original
            path_for_ocr = path

        from modules.ocr_mlkit_android import MlkitOCR
        if self._ocr is None:
            self._ocr = MlkitOCR()

        self.ocr_text = "Recognizing..."
        self._ocr.recognize_file(path_for_ocr, self._on_ocr_done)

    def _on_ocr_done(self, ok, text_or_err):
        if not ok:
            self.ocr_text = f"OCR error: {text_or_err}"
            return
        txt = text_or_err.strip()
        # normalize whitespace
        txt = "\n".join([line.strip() for line in txt.splitlines() if line.strip()])
        self.ocr_text = txt if txt else "(No text found)"

    def send_to_chat(self):
        from kivymd.app import MDApp
        app = MDApp.get_running_app()
        if self.ocr_text.strip():
            app.inject_text_to_chat(self.ocr_text.strip(), auto_send=False)
