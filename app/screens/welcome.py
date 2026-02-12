import os, sys

from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDIconButton, MDFillRoundFlatButton

from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.utils import platform

# get path details
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
    favicon = os.path.join(base_path, "data/images/favicon.png")
else:
    # Running in a normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))
    favicon = os.path.abspath(os.path.join(base_path, "..", "data/images/favicon.png"))

# KV moved to app/screens/welcome.kv

class WelcomeScreen(MDScreen):
    fav_path = StringProperty()
    top_pad = NumericProperty(0)
    bottom_pad = NumericProperty(0)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'welcome_screen'
        self.fav_path = favicon
        if platform == "android":
            try:
                from importlib import import_module
                get_height_of_bar = import_module("android.display_cutout").get_height_of_bar
                self.top_pad = int(get_height_of_bar('status'))
                self.bottom_pad = int(get_height_of_bar('navigation'))
            except Exception as e:
                print(f"Failed android 15 padding: {e}")
                self.top_pad = 32
                self.bottom_pad = 48
