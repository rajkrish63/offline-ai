from kivy.utils import platform
from kivy.clock import Clock

if platform == "android":
    from jnius import autoclass, PythonJavaClass, java_method


class _OnSuccess(PythonJavaClass):
    __javainterfaces__ = ['com/google/android/gms/tasks/OnSuccessListener']
    __javacontext__ = 'app'

    def __init__(self, cb):
        super().__init__()
        self.cb = cb

    @java_method('(Ljava/lang/Object;)V')
    def onSuccess(self, result):
        # result: com.google.mlkit.vision.text.Text
        try:
            text = result.getText()
            Clock.schedule_once(lambda dt: self.cb(True, str(text)))
        except Exception as e:
            Clock.schedule_once(lambda dt: self.cb(False, str(e)))


class _OnFailure(PythonJavaClass):
    __javainterfaces__ = ['com/google/android/gms/tasks/OnFailureListener']
    __javacontext__ = 'app'

    def __init__(self, cb):
        super().__init__()
        self.cb = cb

    @java_method('(Ljava/lang/Exception;)V')
    def onFailure(self, e):
        Clock.schedule_once(lambda dt: self.cb(False, str(e)))


class MlkitOCR:
    def __init__(self):
        if platform != "android":
            raise RuntimeError("MlkitOCR is Android-only.")

        self.PythonActivity = autoclass('org.kivy.android.PythonActivity')
        self.InputImage = autoclass('com.google.mlkit.vision.common.InputImage')
        self.TextRecognition = autoclass('com.google.mlkit.vision.text.TextRecognition')
        self.TextRecognizerOptions = autoclass('com.google.mlkit.vision.text.latin.TextRecognizerOptions')

        self.context = self.PythonActivity.mActivity
        self.recognizer = self.TextRecognition.getClient(self.TextRecognizerOptions.DEFAULT_OPTIONS)

    def recognize_file(self, image_path: str, done_cb):
        """
        done_cb(ok: bool, text_or_error: str)
        """
        try:
            Uri = autoclass('android.net.Uri')
            File = autoclass('java.io.File')
            uri = Uri.fromFile(File(image_path))
            image = self.InputImage.fromFilePath(self.context, uri)

            task = self.recognizer.process(image)
            task.addOnSuccessListener(_OnSuccess(done_cb))
            task.addOnFailureListener(_OnFailure(done_cb))

        except Exception as e:
            print("MLKIT OCR ERROR:", e)
            Clock.schedule_once(lambda dt: done_cb(False, str(e)))
