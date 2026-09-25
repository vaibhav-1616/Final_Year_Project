# VIRTUAL CURSOR - MAIN INTEGRATION FILE

import time

from hand_tracker import HandTracker
from gestures import GestureEngine
from cursor_controller import CursorController
from visualizer import Visualizer


def execute_action(action_data, cursor):

    action = action_data.get("action")

    if action == "click":

        cursor.left_click()

    elif action == "double_click":

        cursor.double_click()

    elif action == "right_click":

        cursor.right_click()

    elif action == "mouse_down":

        cursor.mouse_down()

    elif action == "mouse_up":

        cursor.mouse_up()

    elif action == "scroll":

        cursor.scroll(
            action_data.get("amount", 0)
        )


def main():

    print("==============================================")
    print("       AI-DRIVEN TOUCHLESS VIRTUAL CURSOR")
    print("==============================================")
    print("Starting system...")
    print()
    print("Gestures:")
    print("Index finger       -> Cursor movement")
    print("Thumb + Index      -> Left click")
    print("Rapid second pinch -> Double click")
    print("Held pinch         -> Drag")
    print("Thumb + Middle     -> Right click")
    print("Index + Middle     -> Scroll")
    print("Closed Fist        -> Pause / Resume")
    print()
    print("Press 'x' to exit.")
    print("==============================================")

    tracker = HandTracker()
    gesture_engine = GestureEngine()
    cursor = CursorController()
    visualizer = Visualizer()

    visualizer.initialize_mediapipe_draw(
        tracker.mp_draw,
        tracker.mp_hands
    )

    prev_frame_time = 0

    try:

        while True:

            success, frame, hand_data = (
                tracker.get_frame()
            )

            if not success:

                print(
                    "Unable to read webcam frame."
                )

                break

            height, width, _ = frame.shape

            # ------------------------------------------------
            # Gesture processing
            # ------------------------------------------------

            if hand_data is not None:

                index_x, index_y = (
                    hand_data["index"]
                )

                raw_screen_x, raw_screen_y = (
                    cursor.map_to_screen(
                        index_x,
                        index_y,
                        width,
                        height
                    )
                )

                cursor.initialize_position(
                    raw_screen_x,
                    raw_screen_y
                )

                action_data = (
                    gesture_engine.process(
                        hand_data
                    )
                )

                execute_action(
                    action_data,
                    cursor
                )

                if action_data.get(
                    "move_cursor"
                ):

                    cursor.move_smoothly(
                        raw_screen_x,
                        raw_screen_y
                    )

            else:

                action_data = (
                    gesture_engine.handle_no_hand()
                )

                execute_action(
                    action_data,
                    cursor
                )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            current_time = time.time()

            if prev_frame_time:

                fps = 1 / (
                    current_time
                    - prev_frame_time
                )

            else:

                fps = 0

            prev_frame_time = current_time

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

            frame = visualizer.draw(
                frame,
                hand_data,
                gesture_engine.state,
                fps,
                gesture_engine.paused
            )

            visualizer.show(frame)

            if visualizer.should_exit():

                break

    except KeyboardInterrupt:

        print(
            "\nInterrupted with Ctrl+C."
        )

    finally:

        # Safety: release mouse if dragging
        if gesture_engine.state == "DRAGGING":

            cursor.mouse_up()

        tracker.release()
        visualizer.close()

        print(
            "Virtual Cursor closed safely."
        )


if __name__ == "__main__":

    main()