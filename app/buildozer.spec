[app]

title = OnLLM - Offline AI Chatbot
package.name = onllm
package.domain = in.daslearning

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,ttf,json

source.exclude_dirs = tests, logs, bin, build, dist, patches, .venv, venv, env, .env, p4a_local_recipes

version.regex = __version__ = ['"](.*)['"]
version.filename = %(source.dir)s/main.py


# -----------------------------
# APP REQUIREMENTS
# -----------------------------

requirements = kivy==2.3.1, kivymd==1.2.0, certifi, idna, charset_normalizer, urllib3, pyjnius, android, m2r2, docutils, mistune==0.8.4, filetype, pygments, pillow, requests, coloredlogs, flatbuffers, packaging, numpy1, mpmath, protobuf, sympy, onnxruntime_np1, tokenizers, docx2txt, pypdf, camera4kivy


# -----------------------------
# ICON + SPLASH
# -----------------------------

icon.filename = %(source.dir)s/data/images/favicon.png
presplash.filename = %(source.dir)s/data/images/favicon.png


# -----------------------------
# SCREEN SETTINGS
# -----------------------------

orientation = portrait
fullscreen = 0


# -----------------------------
# ANDROID SETTINGS
# -----------------------------

android.api = 35
android.minapi = 24
android.ndk = 28c

android.archs = arm64-v8a

android.allow_backup = True
android.display_cutout = shortEdges

android.accept_sdk_license = True


# -----------------------------
# PERMISSIONS (SAFE FOR ANDROID 13+)
# -----------------------------

android.permissions = android.permission.INTERNET, android.permission.CAMERA, android.permission.RECORD_AUDIO


# -----------------------------
# ML KIT + VOSK
# -----------------------------

android.gradle_dependencies = com.google.mlkit:text-recognition:16.0.1, com.alphacephei:vosk-android:0.3.75
android.enable_androidx = True
android.gradle_repositories = google(), mavenCentral()


# -----------------------------
# BUILD SETTINGS
# -----------------------------

android.release_artifact = apk


# -----------------------------
# PYTHON FOR ANDROID SETTINGS
# -----------------------------

p4a.url = https://github.com/daslearning-org/p4a-unofficial.git
p4a.branch = numpy2
android.copy_libs = 1
