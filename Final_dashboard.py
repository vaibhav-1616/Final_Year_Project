# Actual_Dashboard.py file:
# AI-DRIVEN TOUCHLESS VIRTUAL CURSOR DASHBOARD (streamlit based)

import time
import threading

import cv2
import streamlit as st

from hand_tracker import HandTracker
from gestures import GestureEngine
from cursor_controller import CursorController
from visualizer import Visualizer

from PIL import Image
import numpy as np
import base64
from pathlib import Path

from config import (
    CAM_WIDTH,
    CAM_HEIGHT,
    FRAME_REDUCTION,
    SMOOTHING_FACTOR,
    CURSOR_DEADBAND,
    CLICK_DISTANCE_THRESHOLD,
    RIGHT_CLICK_DISTANCE_THRESHOLD,
    SCROLL_DISTANCE_THRESHOLD,
    CLICK_HOLD_TIME,
    DRAG_HOLD_TIME,
    DOUBLE_CLICK_WINDOW,
    SCROLL_DEADBAND,
    SCROLL_MULTIPLIER,
    FIST_HOLD_TIME,
    SAFE_MODE,
    STATE_IDLE,
    STATE_LEFT_HOLD,
    STATE_DRAGGING,
    STATE_RIGHT_HOLD,
    STATE_SCROLLING,
    STATE_PAUSED,
)

# PAGE

st.set_page_config(page_title="AI Virtual Cursor", page_icon="🖐️", layout="wide")


# SHARED DATA


class DashboardData:

    def __init__(self):
        self.lock = threading.Lock()

        self.frame = None

        self.running = False
        self.hand_detected = False

        self.state = STATE_IDLE
        self.paused = False

        self.left_clicks = 0
        self.double_clicks = 0
        self.right_clicks = 0
        self.drag_actions = 0
        self.scroll_events = 0
        self.pause_toggles = 0


# SYSTEM


class DashboardSystem:

    def __init__(self):

        self.data = DashboardData()

        self.tracker = None
        self.gesture_engine = None
        self.cursor = None
        self.visualizer = None

        self.thread = None
        self.stop_event = threading.Event()

    # START

    def start(self):

        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_event.clear()

        with self.data.lock:

            # Reset session status
            self.data.running = True
            self.data.hand_detected = False
            self.data.state = STATE_IDLE
            self.data.paused = False

            # Reset session statistics
            self.data.left_clicks = 0
            self.data.double_clicks = 0
            self.data.right_clicks = 0
            self.data.drag_actions = 0
            self.data.scroll_events = 0
            self.data.pause_toggles = 0

            # Clear previous camera frame
            self.data.frame = None

        self.thread = threading.Thread(target=self.run, daemon=True)

        self.thread.start()

    # STOP

    def stop(self):

        self.stop_event.set()

        with self.data.lock:
            self.data.running = False
            self.data.hand_detected = False
            self.data.state = STATE_IDLE
            self.data.paused = False

    # MAIN CAMERA LOOP

    def run(self):

        try:

            # Create actual project components
            self.tracker = HandTracker()
            self.gesture_engine = GestureEngine()
            self.cursor = CursorController()
            self.visualizer = Visualizer()

            self.visualizer.initialize_mediapipe_draw(
                self.tracker.mp_draw, self.tracker.mp_hands
            )

            prev_frame_time = 0

            while not self.stop_event.is_set():

                # ------------------------------------------------
                # GET CAMERA FRAME
                # ------------------------------------------------

                success, frame, hand_data = self.tracker.get_frame()

                if not success or frame is None:
                    continue

                height, width, _ = frame.shape

                # ------------------------------------------------
                # HAND DETECTED
                # ------------------------------------------------

                if hand_data is not None:

                    with self.data.lock:
                        self.data.hand_detected = True

                    index_x, index_y = hand_data["index"]

                    # Convert camera position to screen position
                    raw_x, raw_y = self.cursor.map_to_screen(
                        index_x, index_y, width, height
                    )

                    self.cursor.initialize_position(raw_x, raw_y)

                    # ------------------------------------------------
                    # GESTURE ENGINE
                    # ------------------------------------------------

                    old_paused = self.gesture_engine.paused

                    action_data = self.gesture_engine.process(hand_data)

                    new_paused = self.gesture_engine.paused

                    # Pause / resume counter
                    if old_paused != new_paused:

                        with self.data.lock:
                            self.data.pause_toggles += 1

                    # ------------------------------------------------
                    # EXECUTE ACTION
                    # ------------------------------------------------

                    action = action_data.get("action")

                    if action == "click":

                        self.cursor.left_click()

                        with self.data.lock:
                            self.data.left_clicks += 1

                    elif action == "double_click":

                        self.cursor.double_click()

                        with self.data.lock:
                            self.data.double_clicks += 1

                    elif action == "right_click":

                        self.cursor.right_click()

                        with self.data.lock:
                            self.data.right_clicks += 1

                    elif action == "mouse_down":

                        self.cursor.mouse_down()

                        with self.data.lock:
                            self.data.drag_actions += 1

                    elif action == "mouse_up":

                        self.cursor.mouse_up()

                    elif action == "scroll":

                        self.cursor.scroll(action_data.get("amount", 0))

                        with self.data.lock:
                            self.data.scroll_events += 1

                    # CURSOR MOVEMENT

                    if action_data.get("move_cursor"):

                        self.cursor.move_smoothly(raw_x, raw_y)

                # NO HAND

                else:

                    with self.data.lock:
                        self.data.hand_detected = False

                    action_data = self.gesture_engine.handle_no_hand()

                    if action_data.get("action") == "mouse_up":
                        self.cursor.mouse_up()

                # FPS

                current_time = time.time()

                if prev_frame_time:
                    fps = 1 / (current_time - prev_frame_time)
                else:
                    fps = 0

                prev_frame_time = current_time

                # DRAW OVERLAY (landmarks, state, FPS, pause banner)

                frame = self.visualizer.draw(
                    frame,
                    hand_data,
                    self.gesture_engine.state,
                    fps,
                    self.gesture_engine.paused,
                )

                # PUBLISH VISUALIZED FRAME

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                with self.data.lock:

                    self.data.frame = rgb_frame

                    self.data.state = self.gesture_engine.state

                    self.data.paused = self.gesture_engine.paused

        except Exception as e:

            print(f"Dashboard error: {e}")

        finally:

            # Safety: release mouse if dragging
            if (
                self.gesture_engine is not None
                and self.gesture_engine.state == STATE_DRAGGING
            ):
                self.cursor.mouse_up()

            # Release camera
            if self.tracker is not None:
                self.tracker.release()

            with self.data.lock:

                self.data.running = False
                self.data.hand_detected = False
                self.data.state = STATE_IDLE
                self.data.paused = False

                self.data.frame = None


# SESSION STATE

if "system" not in st.session_state:
    st.session_state.system = DashboardSystem()

system = st.session_state.system
data = system.data


# PAGE STYLING

st.markdown(
    """
<style>
    .stApp {
        background-color: #0e1117;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    h1, h2, h3 {
        color: #f1f5f9;
    }

    .hero {
        padding: 42px 38px;
        border: 1px solid #293548;
        border-radius: 14px;
        background: linear-gradient(110deg, #172337, #111827);
        margin-bottom: 22px;
    }

    .hero-label {
        color: #60a5fa;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 38px;
        font-weight: 700;
        line-height: 1.25;
        margin: 12px 0;
        padding-bottom: 8px;
    }

    .hero-description {
        color: #cbd5e1;
        font-size: 16px;
        line-height: 1.8;
        max-width: 760px;
    }

    .section-label {
        color: #60a5fa;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .info-card {
        background: #151c28;
        border: 1px solid #293548;
        border-radius: 10px;
        padding: 20px;
        min-height: 135px;
        margin-bottom: 28px;
    }

    .info-card .card-title {
        color: #f1f5f9;
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .info-card p {
        color: #aebbd0;
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .step-number {
        color: #60a5fa;
        font-size: 13px;
        font-weight: 700;
    }

    .step-card {
        background: #151c28;
        border: 1px solid #293548;
        border-radius: 10px;
        padding: 18px;
        min-height: 170px;          
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }

    .step-card h3 {
        font-size: 16px;
        margin: 8px 0;
    }

    .step-card p {
        color: #aebbd0;
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 0;
    }

    .notice {
        background: #28251a;
        border: 1px solid #62532a;
        border-radius: 8px;
        padding: 14px 18px;
        color: #e8d9a6;
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 28px;
        margin-top: 16px;
    }

    .footer {
        border-top: 1px solid #293548;
        margin-top: 45px;
        padding: 25px 0 10px;
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.8;
    }

    div.stButton > button {
        border-radius: 8px;
        min-height: 44px;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background: #151c28;
        border: 1px solid #293548;
        padding: 13px;
        border-radius: 9px;
    }

    a {
        color: #60a5fa !important;
        text-decoration: none;
    }

    hr {
        border-color: #293548;
    }

    /* Hide automatic heading anchor icons */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# HERO / HOME

st.html("""
<div class="hero">
    <div class="hero-label">ASSISTIVE TECHNOLOGY PROJECT</div>

    <div class="hero-title">
        AI-Driven Touchless<br>
        Human-Computer Interaction
    </div>

    <div class="hero-description">
        An assistive human-computer interaction system that
        uses webcam-based hand tracking and gestures to perform
        common mouse operations without physical contact.
    </div>
</div>
""")


# PAGE NAVIGATION

st.html("""
<style>
    .nav-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 12px;
        padding: 10px 0 24px 0;
    }

    .nav-item {
        display: inline-block;
        color: #94a3b8;
        background: #151c28;
        border: 1px solid #293548;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        text-decoration: none;
        transition: background 0.2s ease,
                    color 0.2s ease,
                    border-color 0.2s ease;
    }

    .nav-item:hover {
        color: #f8fafc;
        background: #1e293b;
        border-color: #60a5fa;
        text-decoration: none;
    }

    .nav-item:focus-visible {
        outline: 2px solid #60a5fa;
        outline-offset: 3px;
    }
</style>

<div class="nav-container">
    <a href="#about" class="nav-item">About</a>
    <a href="#instructions" class="nav-item">How to Use</a>
    <a href="#live-console" class="nav-item">Live Application</a>
    <a href="#gesture-guide" class="nav-item">Gesture Guide</a>
    <a href="#technical-details" class="nav-item">Technical Details</a>
</div>
""")


# ABOUT THE PROJECT

st.markdown('<div id="about"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">01 / Overview</div>', unsafe_allow_html=True)

st.header("About the Application")

st.write(
    "The AI-Driven Touchless HCI System is designed to provide "
    "a hands-free way of interacting with a computer. It uses "
    "computer vision and hand gesture recognition to translate "
    "hand movements into mouse operations."
)

about1, about2, about3 = st.columns(3)

with about1:
    st.markdown(
        """
    <div class="info-card">
        <div class="card-title">Touchless Interaction</div>
        <p>Control computer operations using hand gestures
        instead of a physical mouse.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with about2:
    st.markdown(
        """
    <div class="info-card">
        <div class="card-title">Real-Time Tracking</div>
        <p>Use a webcam and hand landmark detection to
        track hand movements.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with about3:
    st.markdown(
        """
    <div class="info-card">
        <div class="card-title">Assistive Purpose</div>
        <p>Explore an alternative interaction method that
        may help users with physical interaction difficulties.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# HOW TO USE

st.markdown('<div id="instructions"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="section-label">02 / Getting Started</div>', unsafe_allow_html=True
)

st.header("How to Use")

st.write(
    "Follow these steps to start interacting with your computer "
    "through hand gestures."
)

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown(
        """
    <div class="step-card">
        <div class="step-number">STEP 01</div>
        <h3>Position Your Webcam</h3>
        <p>Place your webcam in front of you. Ensure your hand
        is visible and the surrounding area has sufficient lighting.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with row1_col2:
    st.markdown(
        """
    <div class="step-card">
        <div class="step-number">STEP 02</div>
        <h3>Start the System</h3>
        <p>Go to the live application section and click
        Start System to activate the camera and hand tracking.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown(
        """
    <div class="step-card">
        <div class="step-number">STEP 03</div>
        <h3>Use Hand Gestures</h3>
        <p>Move your index finger to control the cursor and
        use the supported gestures for mouse operations.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with row2_col2:
    st.markdown(
        """
    <div class="step-card">
        <div class="step-number">STEP 04</div>
        <h3>Stop When Finished</h3>
        <p>Click Stop System to stop tracking and release
        the webcam.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="notice">
    <strong>Safety note:</strong> This application can control
    your actual computer mouse. Save your work and close
    sensitive applications before starting the system.
</div>
""",
    unsafe_allow_html=True,
)


# LIVE APPLICATION

st.markdown('<div id="live-console"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="section-label">03 / Live Workspace</div>', unsafe_allow_html=True
)

st.header("Live Control Dashboard")

st.write(
    "Start the system to activate real-time hand tracking "
    "and gesture-based cursor control."
)

control1, control2, control3 = st.columns([1, 1, 2])

with control1:
    if st.button("▶  Start System", use_container_width=True):
        system.start()

with control2:
    if st.button("■  Stop System", use_container_width=True):
        system.stop()

with control3:
    with data.lock:
        running = data.running

    if running:
        st.success("SYSTEM RUNNING")
    else:
        st.info("SYSTEM STOPPED")


# LIVE CAMERA AND STATUS

@st.fragment(run_every=0.20)
def live_monitor():

    camera_col, status_col = st.columns([2.2, 1])

    with camera_col:
        with data.lock:
            frame = data.frame

        if frame is not None:

            st.image(
                frame,
                channels="RGB",
                use_container_width=True
            )

        else:

            st.info(
                "Start the system to activate "
                "the camera and hand tracker."
            )

    with status_col:

        st.subheader("System Status")

        with data.lock:
            state = data.state
            paused = data.paused
            hand = data.hand_detected

            left = data.left_clicks
            double = data.double_clicks
            right = data.right_clicks
            drag = data.drag_actions
            scroll = data.scroll_events
            pause = data.pause_toggles

        if not running:
            st.info("SYSTEM STOPPED")

        elif paused:
            st.warning("PAUSED")

        elif state == STATE_DRAGGING:
            st.error("DRAGGING")

        elif state == STATE_LEFT_HOLD:
            st.warning("LEFT HOLD")

        elif state == STATE_RIGHT_HOLD:
            st.warning("RIGHT HOLD")

        elif state == STATE_SCROLLING:
            st.info("SCROLLING")

        else:
            st.success("🟢 IDLE")

        if hand:
            st.success("🖐 HAND DETECTED")
        else:
            st.warning("NO HAND DETECTED")

        st.divider()

        st.markdown("**Interaction Statistics**")

        m1, m2 = st.columns(2)

        m1.metric("Left Click", left)
        m2.metric("Double Click", double)

        m1.metric("Right Click", right)
        m2.metric("Drag", drag)

        m1.metric("Scroll", scroll)
        m2.metric("Pause / Resume", pause)


live_monitor()


# GESTURE GUIDE

st.markdown('<div id="gesture-guide"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="section-label">04 / Interaction Reference</div>',
    unsafe_allow_html=True,
)

st.header("Gesture Guide")

st.write(
    "The following gestures are supported by the current "
    "gesture recognition implementation."
)

st.markdown(
    """
<style>
    [data-testid="stMarkdownContainer"] table {
        border-collapse: collapse;
        width: 100%;
        border: 1px solid #475569;
    }

    [data-testid="stMarkdownContainer"] th,
    [data-testid="stMarkdownContainer"] td {
        border: 1px solid #475569 !important;
        padding: 12px 14px;
    }

    [data-testid="stMarkdownContainer"] th {
        background-color: #263449;
        color: #ffffff;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("""
| Hand Gesture | Computer Action |
|:---|:---|
| Normal hand tracking ☝️ | Move the cursor using the index fingertip |
| Thumb + Index finger pinch 🤏 | Left click |
| Second pinch within the double-click time window 🤏 | Double click |
| Hold Thumb + Index finger pinch 🤏 | Initiate and maintain drag |
| Thumb + Middle finger pinch 🤏 | Right click |
| Bring Index + Middle fingers close and move vertically ✌️ | Scroll up or down |
| Hold a closed fist ✊ | Toggle Pause / Resume |
""")


# TECHNICAL DETAILS

st.markdown('<div id="technical-details"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="section-label">05 / Technical Information</div>',
    unsafe_allow_html=True,
)

st.header("System Configuration")


with st.expander("View System Parameters"):

    st.html("""
    <style>
        .parameter-list {
            width: 100%;
        }

        .parameter-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 12px 4px;
            border-bottom: 1px solid #334155;
        }

        .parameter-label {
            color: #AAB6C5;
            font-size: 14px;
        }

        .parameter-value {
            color: #F8FAFC;
            font-size: 14px;
            font-weight: 600;
            text-align: right;
        }

        .parameter-row:last-child {
            border-bottom: none;
        }
    </style>
    """)

    def parameter_row(label, value):
        return f"""
        <div class="parameter-row">
            <span class="parameter-label">{label}</span>
            <span class="parameter-value">{value}</span>
        </div>
        """

    c1, c2 = st.columns(2)

    with c1:
        left_parameters = [
            ("Camera Resolution", f"{CAM_WIDTH} × {CAM_HEIGHT}"),
            ("Frame Reduction", f"{FRAME_REDUCTION}px"),
            ("Smoothing Factor", SMOOTHING_FACTOR),
            ("Cursor Deadband", f"{CURSOR_DEADBAND}px"),
            ("Click Threshold", f"{CLICK_DISTANCE_THRESHOLD}px"),
            ("Right Click Threshold", f"{RIGHT_CLICK_DISTANCE_THRESHOLD}px"),
            ("Scroll Threshold", f"{SCROLL_DISTANCE_THRESHOLD}px"),
        ]

        st.html(
            '<div class="parameter-list">'
            + "".join(parameter_row(k, v) for k, v in left_parameters)
            + "</div>"
        )

    with c2:
        right_parameters = [
            ("Click Hold Time", f"{CLICK_HOLD_TIME}s"),
            ("Drag Hold Time", f"{DRAG_HOLD_TIME}s"),
            ("Double Click Window", f"{DOUBLE_CLICK_WINDOW}s"),
            ("Scroll Deadband", f"{SCROLL_DEADBAND}px"),
            ("Scroll Multiplier", SCROLL_MULTIPLIER),
            ("Fist Hold Time", f"{FIST_HOLD_TIME}s"),
            ("Safe Mode", "Enabled" if SAFE_MODE else "Disabled"),
        ]

        st.html(
            '<div class="parameter-list">'
            + "".join(parameter_row(k, v) for k, v in right_parameters)
            + "</div>"
        )


with st.expander("System Architecture"):

    image_path = Path(__file__).parent / "system_architecture.png"

    if image_path.exists():

        image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        st.html(f"""
            <div style="width: 100%; overflow: hidden; border-radius: 10px;">
                <img
                    src="data:image/png;base64,{image_base64}"
                    alt="System Architecture Diagram"
                    style="width: 100%; height: auto; display: block;"
                />
            </div>
        """)

    else:
        st.error("System Architecture image not found.")


with st.expander("Technologies Used"):

    st.markdown("""
    | Technology | Role |
    |:---|:---|
    | Python | Main programming language |
    | OpenCV | Webcam access and image processing |
    | MediaPipe | Hand landmark detection |
    | NumPy | Numerical computations and array operations |
    | PyAutoGUI | Mouse control |
    | Streamlit | Dashboard interface |
    """)


# FOOTER

st.html("""
<style>
    .footer-container {
        border-top: 1px solid #293548;
        margin-top: 55px;
        padding: 30px 10px 18px 10px;
        text-align: center;
    }

    .footer-title {
        color: #F1F5F9;
        font-size: 20px;
        font-weight: 650;
        letter-spacing: 0.2px;
        margin-bottom: 7px;
    }

    .footer-subtitle {
        color: #60A5FA;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 16px;
    }

    .footer-description {
        color: #CBD5E1;
        font-size: 13px;
        line-height: 1.7;
        max-width: 650px;
        margin: 0 auto 20px auto;
    }

    .footer-tech-label {
        color: #64748B;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
    }

    .footer-tech {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 9px;
        margin-bottom: 22px;
    }

    .footer-tech span {
        color: #CBD5E1;
        background: #1E293B;
        border: 1px solid #334155;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
    }

    .footer-bottom {
        border-top: 1px solid #293548;
        padding-top: 16px;
        color: #64748B;
        font-size: 11px;
    }
</style>

<div class="footer-container">

    <div class="footer-title">
        AI-Driven Touchless HCI System
    </div>

    <div class="footer-subtitle">
        Assistive Technology Project
    </div>

    <div class="footer-description">
        Enabling intuitive, touch-free computer interaction
        through real-time hand gesture recognition and
        computer vision.
    </div>

    <div class="footer-tech-label">POWERED BY</div>

    <div class="footer-tech">
        <span>Python</span>
        <span>OpenCV</span>
        <span>MediaPipe</span>
        <span>PyAutoGUI</span>
        <span>Streamlit</span>
    </div>

    <div class="footer-bottom">
        Touchless interaction through computer vision and hand gestures.
    </div>

</div>
""")