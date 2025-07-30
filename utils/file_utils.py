import tempfile
import os

def save_uploaded_file(uploaded_file):
    _, suffix = os.path.splitext(uploaded_file.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name
