# main.py — YOLOv8 Detection Dashboard
# Fix #6: removed all unused imports (cv2, np, tempfile, PIL, random, defaultdict, os)

import streamlit as st
import warnings

warnings.filterwarnings('ignore')

# MUST be the first Streamlit command
st.set_page_config(
    page_title="YOLOv8 Detection Dashboard",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None},
)

from ui_components import load_css_from_file
from session_state import init_session_state
from sidebar import render_sidebar
from tabs import render_tabs

# Load CSS
load_css_from_file()

# Fix #23: title colour updated to match dark theme design system (#F1F5F9 heading)
st.markdown("""
<div style="padding-top:0;margin-top:0;">
    <h1 style="font-size:1.75rem;color:#F1F5F9;font-weight:700;
               margin-bottom:0.25rem;padding-top:0;">
        YOLOv8 Detection Dashboard
    </h1>
    <div style="color:#94A3B8;font-weight:400;line-height:1.5;margin-bottom:1.5rem;">
        Real-time object detection and tracking. Upload a video or use your webcam
        with advanced multi-class object tracking.
    </div>
</div>
""", unsafe_allow_html=True)

# Initialise session state (pre-loads model silently)
init_session_state()

# Render sidebar and main content
render_sidebar()
render_tabs()
