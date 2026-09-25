# AI-DRIVEN TOUCHLESS VIRTUAL CURSOR STREAMLIT MONITORING DASHBOARD

# command to run : streamlit run Final_implementation\testing.py

import time
import threading
from collections import deque

import cv2
import streamlit as st

from hand_tracker import HandTracker
from gestures import GestureEngine
from cursor_controller import CursorController

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
    STATE_PAUSED
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Virtual Cursor Dashboard",
    page_icon="🖐️",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "system" not in st.session_state:
    st.session_state.system = None

if "worker" not in st.session_state:
    st.session_state.worker = None


# ============================================================
# SHARED SYSTEM DATA
# ============================================================

class DashboardData:

    def __init__(self):
        self.lock = threading.Lock()

        self.frame = None

        self.running = False

        self.state = STATE_IDLE
        self.paused = False

        self.fps = 0
        self.fps_history = deque(maxlen=120)

        self.start_time = None

        self.left_clicks = 0
        self.double_clicks = 0
        self.right_clicks = 0
        self.drag_actions = 0
        self.scroll_events = 0
        self.pause_toggles = 0

        self.event_log = deque(maxlen=20)

        self.hand_detected = False

    def add_event(self, event):
        timestamp = time.strftime("%H:%M:%S")
        self.event_log.appendleft(f"{timestamp} - {event}")


# ============================================================
# BACKGROUND VIRTUAL CURSOR
# ============================================================

class DashboardSystem:

    def __init__(self):
        self.data = DashboardData()

        self.tracker = None
        self.gesture_engine = None
        self.cursor = None

        self.thread = None

        self.stop_event = threading.Event()

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_event.clear()

        self.data.running = True
        self.data.start_time = time.time()

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.data.running = False

    def run(self):
        try:
            self.tracker = HandTracker()
            self.gesture_engine = GestureEngine()
            self.cursor = CursorController()

            previous_time = time.time()

            while not self.stop_event.is_set():

                success, frame, hand_data = self.tracker.get_frame()

                if not success:
                    self.data.add_event("Webcam frame unavailable")
                    break

                height, width, _ = frame.shape

                # ------------------------------------------------
                # Process hand
                # ------------------------------------------------

                if hand_data is not None:

                    self.data.hand_detected = True

                    index_x, index_y = hand_data["index"]

                    raw_x, raw_y = self.cursor.map_to_screen(
                        index_x, index_y, width, height
                    )

                    self.cursor.initialize_position(raw_x, raw_y)

                    old_paused = self.gesture_engine.paused

                    action_data = self.gesture_engine.process(hand_data)

                    new_paused = self.gesture_engine.paused

                    # ------------------------------------------------
                    # Track pause/resume
                    # ------------------------------------------------

                    if old_paused != new_paused:
                        self.data.pause_toggles += 1

                        if new_paused:
                            self.data.add_event("System paused using fist gesture")
                        else:
                            self.data.add_event("System resumed using fist gesture")

                    # ------------------------------------------------
                    # Execute actions
                    # ------------------------------------------------

                    action = action_data.get("action")

                    if action == "click":
                        self.cursor.left_click()
                        self.data.left_clicks += 1
                        self.data.add_event("Left click")

                    elif action == "double_click":
                        self.cursor.double_click()
                        self.data.double_clicks += 1
                        self.data.add_event("Double click")

                    elif action == "right_click":
                        self.cursor.right_click()
                        self.data.right_clicks += 1
                        self.data.add_event("Right click")

                    elif action == "mouse_down":
                        self.cursor.mouse_down()
                        self.data.drag_actions += 1
                        self.data.add_event("Drag started")

                    elif action == "mouse_up":
                        self.cursor.mouse_up()
                        self.data.add_event("Drag ended")

                    elif action == "scroll":
                        self.cursor.scroll(action_data.get("amount", 0))
                        self.data.scroll_events += 1

                    # ------------------------------------------------
                    # Cursor movement
                    # ------------------------------------------------

                    if action_data.get("move_cursor"):
                        self.cursor.move_smoothly(raw_x, raw_y)

                else:
                    self.data.hand_detected = False

                    action_data = self.gesture_engine.handle_no_hand()

                    if action_data.get("action") == "mouse_up":
                        self.cursor.mouse_up()

                # ------------------------------------------------
                # FPS
                # ------------------------------------------------

                current_time = time.time()
                elapsed = current_time - previous_time
                fps = 1 / elapsed if elapsed > 0 else 0
                previous_time = current_time

                # ------------------------------------------------
                # Publish frame + stats to the shared, locked state
                # ------------------------------------------------

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                with self.data.lock:
                    self.data.fps = fps
                    self.data.fps_history.append(fps)
                    self.data.state = self.gesture_engine.state
                    self.data.paused = self.gesture_engine.paused
                    self.data.frame = rgb_frame

            # End loop

        except Exception as e:
            self.data.add_event(f"System error: {str(e)}")

        finally:
            if (
                self.gesture_engine is not None
                and self.gesture_engine.state == STATE_DRAGGING
            ):
                self.cursor.mouse_up()

            if self.tracker is not None:
                self.tracker.release()

            self.data.running = False


# ============================================================
# HELPERS
# ============================================================

def get_system():
    if st.session_state.system is None:
        st.session_state.system = DashboardSystem()

    return st.session_state.system


def average_fps(data):
    with data.lock:
        values = list(data.fps_history)

    return sum(values) / len(values) if values else 0


def minimum_fps(data):
    with data.lock:
        values = list(data.fps_history)

    return min(values) if values else 0


def maximum_fps(data):
    with data.lock:
        values = list(data.fps_history)

    return max(values) if values else 0


# ============================================================
# HEADER
# ============================================================

st.title("🖐️ AI-Driven Touchless Virtual Cursor")
st.caption("Assistive Human-Computer Interaction System")
st.divider()


# ============================================================
# CONTROL BAR
# ============================================================

system = get_system()

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶ Start System", use_container_width=True):
        system.start()

with col2:
    if st.button("⏹ Stop System", use_container_width=True):
        system.stop()

with col3:
    if system.data.running:
        st.success("🟢 SYSTEM RUNNING")
    else:
        st.warning("⚪ SYSTEM STOPPED")


# ============================================================
# LIVE MONITOR  (auto-refreshing fragment — this is the fix)
# ============================================================
#
# Root cause of the missing feed: the old code rendered st.image()
# only once per full script run, and the "auto refresh" line at the
# bottom (`st.fragment(run_every="1s")(lambda: None)()`) wrapped an
# empty no-op instead of the image code, so nothing ever told
# Streamlit to redraw the frame. Wrapping the actual rendering in a
# real @st.fragment(run_every=...) makes Streamlit re-execute just
# this block on a timer, independent of the rest of the page.

st.subheader("Live System Monitor")


@st.fragment(run_every=0.15)
def render_live_monitor():

    live_col, status_col = st.columns([2, 1])

    with live_col:
        st.markdown("**Live Camera / Hand Tracking**")

        with system.data.lock:
            frame = system.data.frame

        if frame is not None:
            st.image(frame, channels="RGB", use_container_width=True)
        else:
            st.info("Start the system to view the camera feed.")

    with status_col:
        st.markdown("### System Status")

        state = system.data.state

        if state == STATE_PAUSED:
            st.warning("⏸ PAUSED")
        elif system.data.running:
            st.success(f"🟢 {state}")
        else:
            st.info("System stopped")

        if system.data.hand_detected:
            st.success("🖐 Hand detected")
        else:
            st.warning("No hand detected")

        st.metric("Current FPS", f"{system.data.fps:.1f}")


render_live_monitor()


# ============================================================
# PERFORMANCE  (also auto-refreshing)
# ============================================================

st.subheader("Performance Monitor")


@st.fragment(run_every=0.5)
def render_performance():

    p1, p2, p3, p4 = st.columns(4)

    p1.metric("Current FPS", f"{system.data.fps:.1f}")
    p2.metric("Average FPS", f"{average_fps(system.data):.1f}")
    p3.metric("Minimum FPS", f"{minimum_fps(system.data):.1f}")
    p4.metric("Maximum FPS", f"{maximum_fps(system.data):.1f}")

    with system.data.lock:
        fps_values = list(system.data.fps_history)

    if fps_values:
        st.line_chart(fps_values, height=220)


render_performance()


# ============================================================
# SESSION STATISTICS  (also auto-refreshing)
# ============================================================

st.subheader("Session Statistics")


@st.fragment(run_every=0.5)
def render_stats():

    s1, s2, s3, s4 = st.columns(4)

    s1.metric("Left Clicks", system.data.left_clicks)
    s2.metric("Double Clicks", system.data.double_clicks)
    s3.metric("Right Clicks", system.data.right_clicks)
    s4.metric("Drag Actions", system.data.drag_actions)

    s5, s6, s7 = st.columns(3)

    s5.metric("Scroll Events", system.data.scroll_events)
    s6.metric("Pause / Resume", system.data.pause_toggles)

    if system.data.start_time:
        duration = time.time() - system.data.start_time
    else:
        duration = 0

    s7.metric("Session Time", f"{duration:.0f} sec")


render_stats()


# ============================================================
# GESTURE MAPPING
# ============================================================

st.subheader("Gesture Mapping")

st.markdown(
    """
| Gesture | Action |
|---|---|
| ☝️ Index finger | Cursor movement |
| 🤏 Thumb + Index | Left click |
| 🤏 Rapid second pinch | Double click |
| 🤏 Held Thumb + Index | Drag |
| 🤏 Thumb + Middle | Right click |
| ✌️ Index + Middle | Scroll |
| ✊ Fist held for 1 second | Pause / Resume |
"""
)


# ============================================================
# CONFIGURATION
# ============================================================

with st.expander("⚙️ Gesture & System Configuration"):

    c1, c2 = st.columns(2)

    with c1:
        st.write(f"Camera: {CAM_WIDTH} × {CAM_HEIGHT}")
        st.write(f"Active Region Reduction: {FRAME_REDUCTION}px")
        st.write(f"Base Smoothing Factor: {SMOOTHING_FACTOR}")
        st.write(f"Cursor Deadband: {CURSOR_DEADBAND}px")
        st.write(f"Click Threshold: {CLICK_DISTANCE_THRESHOLD}px")
        st.write(f"Right Click Threshold: {RIGHT_CLICK_DISTANCE_THRESHOLD}px")

    with c2:
        st.write(f"Scroll Threshold: {SCROLL_DISTANCE_THRESHOLD}px")
        st.write(f"Click Hold: {CLICK_HOLD_TIME}s")
        st.write(f"Drag Hold: {DRAG_HOLD_TIME}s")
        st.write(f"Double Click Window: {DOUBLE_CLICK_WINDOW}s")
        st.write(f"Scroll Deadband: {SCROLL_DEADBAND}px")
        st.write(f"Fist Hold: {FIST_HOLD_TIME}s")
        st.write(f"Safe Mode: {'Enabled' if SAFE_MODE else 'Disabled'}")


# ============================================================
# ARCHITECTURE
# ============================================================

with st.expander("🏗️ System Architecture"):

    st.code(
        """
                VIRTUAL CURSOR SYSTEM
                         │
                         ▼
                  Webcam Input
                         │
                         ▼
                  Hand Tracker
                  (MediaPipe)
                         │
                         ▼
                 Gesture Engine
                  ┌──────┴──────┐
                  │             │
              FSM Logic     Fist Control
                  │             │
                  └──────┬──────┘
                         ▼
                Cursor Controller
                 ┌───────┴───────┐
                 │               │
             EMA Filter      PyAutoGUI
                 │               │
                 └───────┬───────┘
                         ▼
                   Computer UI


       dashboard.py
             │
             ├── Live Monitoring
             ├── Performance
             ├── Statistics
             ├── Configuration
             └── Event Logging
        """,
        language="text"
    )


# ============================================================
# EVENT LOG  (also auto-refreshing)
# ============================================================

with st.expander("📋 Recent System Events"):

    @st.fragment(run_every=1)
    def render_events():
        if system.data.event_log:
            for event in list(system.data.event_log):
                st.write(f"• {event}")
        else:
            st.info("No events recorded yet.")

    render_events()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Driven Touchless HCI System | "
    "MediaPipe + OpenCV + PyAutoGUI + Streamlit"
)