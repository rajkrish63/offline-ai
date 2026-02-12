# screens/chatbot_screen.py
import sys, os

from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.label import MDLabel
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dropdownitem import MDDropDownItem
from kivymd.uix.button import MDIconButton, MDFillRoundFlatButton, MDFillRoundFlatIconButton, MDFloatingActionButton
from kivymd.uix.behaviors import HoverBehavior

from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import StringProperty, NumericProperty, ObjectProperty, BooleanProperty, ColorProperty
from kivy.utils import platform
from kivy.animation import Animation

# local imports
from .myrst import MyRstDocument

# get path details
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
    noto_font = os.path.join(base_path, "data/fonts/NotoSans-Merged.ttf")
else:
    # Running in a normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))
    noto_font = os.path.abspath(os.path.join(base_path, "..", "data/fonts/NotoSans-Merged.ttf"))

# KV moved to app/screens/chatbot.kv

class TempSpinWait(MDBoxLayout):
    text = StringProperty("")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Fade in animation
        anim = Animation(opacity=1, duration=0.3)
        anim.start(self)

class SidebarItem(MDCard, HoverBehavior):
    text = StringProperty("")
    active = BooleanProperty(False)
    normal_color = ColorProperty((0, 0, 0, 0))
    hover_color = ColorProperty((0.102, 0.102, 0.102, 1))  # #1a1a1a
    active_color = ColorProperty((0.102, 0.102, 0.102, 1))

    def on_kv_post(self, *args):
        super().on_kv_post(*args)
        self.md_bg_color = self.active_color if self.active else self.normal_color

    def on_enter(self):
        if not self.active:
            Animation(md_bg_color=self.hover_color, d=0.12).start(self)

    def on_leave(self):
        if not self.active:
            Animation(md_bg_color=self.normal_color, d=0.12).start(self)

    def on_active(self, *_):
        target = self.active_color if self.active else self.normal_color
        Animation(md_bg_color=target, d=0.12).start(self)

class UsrResp(MDBoxLayout):
    text = StringProperty("")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Slide in from right with fade
        anim = Animation(opacity=1, duration=0.3, t='out_cubic')
        anim.start(self)

class BotTmpResp(MDBoxLayout):
    text = StringProperty("")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Fade in animation
        anim = Animation(opacity=1, duration=0.3)
        anim.start(self)

class BotResp(MDBoxLayout):
    text = StringProperty("")
    given_id = NumericProperty()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Slide in from left with fade
        anim = Animation(opacity=1, duration=0.3, t='out_cubic')
        anim.start(self)

class ChatbotScreen(MDScreen):
    noto_path = StringProperty()
    top_pad = NumericProperty(0)
    bottom_pad = NumericProperty(0)
    def __init__(self, noto=noto_font, **kwargs):
        super().__init__(**kwargs)
        self.name = 'chatbot_screen'
        self.noto_path = noto
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
