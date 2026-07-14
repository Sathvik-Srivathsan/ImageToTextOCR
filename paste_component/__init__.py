import streamlit.components.v1 as components
import os

_component_dir = os.path.join(os.path.dirname(__file__))
_paste_component = components.declare_component("paste_image", path=_component_dir)


def paste_image(key=None):
    return _paste_component(default="", key=key)
