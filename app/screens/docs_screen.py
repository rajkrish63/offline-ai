from kivymd.uix.screen import MDScreen
from kivy.properties import StringProperty


class DocsScreen(MDScreen):
    status = StringProperty("Upload PDF/DOCX/TXT. Then go to Chat and ask with DOC mode ON.")

    def pick_document(self):
        from kivymd.app import MDApp
        app = MDApp.get_running_app()
        app.open_doc_picker(from_screen="docs")

    def go_chat(self):
        from kivymd.app import MDApp
        app = MDApp.get_running_app()
        app.root.current = "chatbot_screen"
