import cv2
import numpy as np
import mediapipe as mp


class FunnyFaceEmojiTask:
    """
    Special Task:
    Neon NPC Emoji Cam.

    This task uses MediaPipe Face Mesh, a pretrained neural-network model,
    to detect facial landmarks in real time.

    It has two emotion modes:

    1. AUTO MODE:
       The expression is estimated from face landmark geometry.

    2. MANUAL DEMO MODE:
       You can press keys to force emotions for a reliable presentation demo.

    Controls:
    A = Auto mode
    H = Happy
    X = Excited
    K = Shocked
    Z = Sleepy
    D = Sad
    C = Crying
    G = Angry
    N = Neutral
    L = Show/hide landmarks
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.show_landmarks = False

        # Auto/manual emotion switching
        self.demo_override = False
        self.forced_expression = "neutral"
        self.last_expression = "neutral"

        # Neon color palette in BGR format
        self.bg_dark = (18, 14, 30)
        self.card_dark = (24, 20, 42)
        self.neon_cyan = (255, 255, 0)
        self.neon_magenta = (255, 60, 200)
        self.neon_purple = (220, 90, 255)
        self.neon_green = (80, 255, 80)
        self.neon_red = (60, 60, 255)
        self.neon_orange = (0, 180, 255)
        self.soft_white = (245, 245, 245)
        self.muted_white = (210, 210, 210)

        self.expression_info = {
            "happy": {
                "title": "HAPPY NPC",
                "message": "Finally understood the lecture!",
                "status": "Confidence: 91% | Stress: 14%",
                "mood": "GOOD",
                "meter": 88,
            },
            "excited": {
                "title": "MAIN CHARACTER MODE",
                "message": "This demo is about to go viral.",
                "status": "Energy: 97% | Confidence: 95%",
                "mood": "HYPED",
                "meter": 98,
            },
            "shocked": {
                "title": "DEADLINE ALERT",
                "message": "Assignment deadline detected!",
                "status": "Panic: 99% | Stability: 3%",
                "mood": "PANIC",
                "meter": 100,
            },
            "sleepy": {
                "title": "SLEEP MODE",
                "message": "Battery low. Coffee required.",
                "status": "Energy: 11% | Focus: 19%",
                "mood": "LOW",
                "meter": 16,
            },
            "sad": {
                "title": "SAD DEBUGGER",
                "message": "It runs... but the result is suspicious.",
                "status": "Motivation: 36% | Hope: 28%",
                "mood": "DOWN",
                "meter": 30,
            },
            "crying": {
                "title": "CRYING COMPILER VICTIM",
                "message": "One bug fixed, three bugs unlocked.",
                "status": "Tears: 92% | Debugging: 99%",
                "mood": "CRITICAL",
                "meter": 24,
            },
            "angry": {
                "title": "ANGRY NPC",
                "message": "Why is this import still failing?!",
                "status": "Anger: 87% | Patience: 8%",
                "mood": "FURIOUS",
                "meter": 22,
            },
            "neutral": {
                "title": "NPC STUDENT",
                "message": "Processing lecture content...",
                "status": "Focus: 50% | Confusion: 50%",
                "mood": "NORMAL",
                "meter": 50,
            },
            "confused": {
                "title": "CONFUSED NPC",
                "message": "Quest updated: survive Computer Vision.",
                "status": "Confusion: 90% | Deadline Fear: 74%",
                "mood": "CONFUSED",
                "meter": 42,
            },
        }

    def set_mode_from_key(self, key):
        """
        Keyboard controls for auto/manual emotion modes.
        """

        if key == ord("l"):
            self.show_landmarks = not self.show_landmarks
            print("Landmarks:", self.show_landmarks)

        elif key == ord("a"):
            self.demo_override = False
            print("Emotion mode: AUTO")

        elif key == ord("h"):
            self.demo_override = True
            self.forced_expression = "happy"
            print("Emotion forced: HAPPY")

        elif key == ord("x"):
            self.demo_override = True
            self.forced_expression = "excited"
            print("Emotion forced: EXCITED")

        elif key == ord("k"):
            self.demo_override = True
            self.forced_expression = "shocked"
            print("Emotion forced: SHOCKED")

        elif key == ord("z"):
            self.demo_override = True
            self.forced_expression = "sleepy"
            print("Emotion forced: SLEEPY")

        elif key == ord("d"):
            self.demo_override = True
            self.forced_expression = "sad"
            print("Emotion forced: SAD")

        elif key == ord("c"):
            self.demo_override = True
            self.forced_expression = "crying"
            print("Emotion forced: CRYING")

        elif key == ord("g"):
            self.demo_override = True
            self.forced_expression = "angry"
            print("Emotion forced: ANGRY")

        elif key == ord("n"):
            self.demo_override = True
            self.forced_expression = "neutral"
            print("Emotion forced: NEUTRAL")

    def process_frame(self, frame):
        """
        Main function called from main.py.
        """

        output = frame.copy()
        h, w, _ = output.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        output = self.draw_top_hud(output)

        if not results.multi_face_landmarks:
            output = self.draw_no_face_state(output)
            return output

        face_landmarks = results.multi_face_landmarks[0]
        points = []

        for landmark in face_landmarks.landmark:
            points.append((int(landmark.x * w), int(landmark.y * h)))

        x, y, box_w, box_h = self.get_face_box(points, w, h)

        if self.demo_override:
            expression = self.forced_expression
        else:
            expression = self.detect_expression(points)

        self.last_expression = expression

        if self.show_landmarks:
            output = self.draw_landmarks(output, points)

        output = self.draw_neon_face_box(output, x, y, box_w, box_h)
        output = self.draw_emoji_overlay(output, x, y, box_w, box_h, expression)
        output = self.draw_npc_bubble(output, x, y, box_w, box_h, expression)
        output = self.draw_status_panel(output, expression)
        output = self.draw_emotion_card(output, expression)

        return output

    def get_face_box(self, points, frame_w, frame_h):
        """
        Create a bounding box around the detected face landmarks.
        """

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min = max(0, min(xs) - 45)
        y_min = max(0, min(ys) - 65)
        x_max = min(frame_w, max(xs) + 45)
        y_max = min(frame_h, max(ys) + 55)

        return x_min, y_min, x_max - x_min, y_max - y_min

    def distance(self, p1, p2):
        """
        Calculate Euclidean distance between two points.
        """

        return np.linalg.norm(np.array(p1) - np.array(p2))

    def detect_expression(self, points):
        """
        Auto expression detection using face landmark ratios.

        Face Mesh does not directly classify real emotions.
        It gives facial geometry, so we estimate expressions using:
        - mouth opening
        - eye opening
        - eyebrow distance
        - mouth curve

        Auto mode is best for:
        happy, excited, shocked, sleepy, neutral/confused.

        Crying, angry, and sad are harder, so manual keys are provided.
        """

        left_mouth = points[61]
        right_mouth = points[291]
        upper_lip = points[13]
        lower_lip = points[14]

        forehead = points[10]
        chin = points[152]

        left_eye_top = points[159]
        left_eye_bottom = points[145]
        right_eye_top = points[386]
        right_eye_bottom = points[374]

        left_brow = points[105]
        right_brow = points[334]
        left_eye_center = points[159]
        right_eye_center = points[386]

        left_corner = points[61]
        right_corner = points[291]

        mouth_center = (
            int((upper_lip[0] + lower_lip[0]) / 2),
            int((upper_lip[1] + lower_lip[1]) / 2),
        )

        mouth_width = max(self.distance(left_mouth, right_mouth), 1)
        mouth_open = self.distance(upper_lip, lower_lip)

        left_eye_open = self.distance(left_eye_top, left_eye_bottom)
        right_eye_open = self.distance(right_eye_top, right_eye_bottom)
        avg_eye_open = (left_eye_open + right_eye_open) / 2

        face_height = max(self.distance(forehead, chin), 1)

        mouth_open_ratio = mouth_open / mouth_width
        eye_open_ratio = avg_eye_open / mouth_width

        brow_left_distance = abs(left_eye_center[1] - left_brow[1]) / face_height
        brow_right_distance = abs(right_eye_center[1] - right_brow[1]) / face_height
        brow_distance = (brow_left_distance + brow_right_distance) / 2

        mouth_corner_avg_y = (left_corner[1] + right_corner[1]) / 2
        mouth_center_y = mouth_center[1]
        mouth_curve = (mouth_corner_avg_y - mouth_center_y) / mouth_width

        # Sleepy: eyes are almost closed
        if eye_open_ratio < 0.045:
            return "sleepy"

        # Shocked: mouth very open and eyes open
        if mouth_open_ratio > 0.30 and eye_open_ratio > 0.070:
            return "shocked"

        # Excited: mouth open, but not shocked
        if mouth_open_ratio > 0.22:
            return "excited"

        # Happy: medium mouth opening / smile-like state
        if mouth_open_ratio > 0.11:
            return "happy"

        # Angry: eyebrows close to eyes
        if brow_distance < 0.045 and eye_open_ratio > 0.045:
            return "angry"

        # Crying checked before sad, otherwise sad catches it first
        if mouth_curve > 0.035 and eye_open_ratio < 0.070:
            return "crying"

        # Sad: mouth corners downward
        if mouth_curve > 0.055:
            return "sad"

        # Neutral
        if 0.060 <= mouth_open_ratio <= 0.10:
            return "neutral"

        return "confused"

    def draw_glow_rect(self, frame, pt1, pt2, color, thickness=2):
        """
        Draw glowing rectangle effect.
        """

        for t, alpha in [(10, 0.06), (6, 0.09), (3, 0.13)]:
            overlay = frame.copy()
            cv2.rectangle(overlay, pt1, pt2, color, t)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.rectangle(frame, pt1, pt2, color, thickness)
        return frame

    def draw_glow_line(self, frame, p1, p2, color):
        """
        Draw glowing line effect.
        """

        for thickness, alpha in [(8, 0.08), (4, 0.12), (2, 0.18)]:
            overlay = frame.copy()
            cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame

    def draw_top_hud(self, frame):
        """
        Draw top neon frontend bar.
        """

        h, w, _ = frame.shape

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 118), self.bg_dark, -1)
        frame = cv2.addWeighted(overlay, 0.84, frame, 0.16, 0)

        frame = self.draw_glow_line(frame, (0, 116), (w, 116), self.neon_cyan)

        mode_text = "AUTO MODE" if not self.demo_override else f"MANUAL: {self.forced_expression.upper()}"

        cv2.putText(
            frame,
            "NEON NPC EMOJI CAM",
            (26, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.92,
            self.soft_white,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Mode: {mode_text} | FaceMesh Neural Network + OpenCV Real-Time Overlay",
            (26, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            self.neon_cyan,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "A Auto | H Happy | X Excited | K Shocked | Z Sleepy | D Sad | C Crying | G Angry | N Neutral | L Landmarks",
            (26, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            self.muted_white,
            1,
            cv2.LINE_AA,
        )

        return frame

    def draw_no_face_state(self, frame):
        """
        Draw message when no face is detected.
        """

        h, w, _ = frame.shape
        x1, y1, x2, y2 = 40, h - 130, 680, h - 45

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.card_dark, -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        frame = self.draw_glow_rect(frame, (x1, y1), (x2, y2), self.neon_magenta, 2)

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (60, h - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            self.neon_magenta,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "NPC is probably making coffee or hiding from the deadline.",
            (60, h - 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            self.soft_white,
            1,
            cv2.LINE_AA,
        )

        return frame

    def draw_landmarks(self, frame, points):
        """
        Draw simplified face landmarks.
        """

        for point in points[::4]:
            cv2.circle(frame, point, 1, self.neon_green, -1)

        return frame

    def draw_neon_face_box(self, frame, x, y, box_w, box_h):
        """
        Draw neon box around face.
        """

        return self.draw_glow_rect(
            frame,
            (x, y),
            (x + box_w, y + box_h),
            self.neon_cyan,
            2,
        )

    def draw_star(self, frame, cx, cy, size, color):
        """
        Draw star shape for excited emoji eyes.
        """

        points = np.array(
            [
                [cx, cy - size],
                [cx + size // 3, cy - size // 3],
                [cx + size, cy - size // 3],
                [cx + size // 2, cy + size // 6],
                [cx + size * 2 // 3, cy + size],
                [cx, cy + size // 2],
                [cx - size * 2 // 3, cy + size],
                [cx - size // 2, cy + size // 6],
                [cx - size, cy - size // 3],
                [cx - size // 3, cy - size // 3],
            ],
            np.int32,
        )

        cv2.fillPoly(frame, [points], color)
        cv2.polylines(frame, [points], True, (0, 0, 0), 2)

    def draw_emoji_overlay(self, frame, x, y, box_w, box_h, expression):
        """
        Draw custom emoji overlay on detected face.
        """

        center_x = x + box_w // 2
        center_y = y + box_h // 2
        radius = int(max(box_w, box_h) * 0.46)

        face_colors = {
            "happy": (0, 225, 255),
            "excited": (40, 240, 255),
            "shocked": (0, 200, 255),
            "sleepy": (210, 220, 255),
            "sad": (255, 190, 90),
            "crying": (255, 200, 100),
            "angry": (60, 90, 255),
            "neutral": (0, 210, 230),
            "confused": (100, 210, 255),
        }

        face_color = face_colors.get(expression, (0, 220, 255))

        # Glow behind emoji
        overlay = frame.copy()
        cv2.circle(overlay, (center_x, center_y), radius + 18, self.neon_magenta, -1)
        frame = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)

        # Shadow
        cv2.circle(frame, (center_x + 8, center_y + 8), radius, (15, 15, 15), -1)

        # Face base
        cv2.circle(frame, (center_x, center_y), radius, face_color, -1)
        cv2.circle(frame, (center_x, center_y), radius, self.soft_white, 3)

        # Highlight
        cv2.circle(
            frame,
            (center_x - radius // 3, center_y - radius // 3),
            radius // 5,
            self.soft_white,
            -1,
        )

        eye_y = center_y - radius // 4
        left_eye_x = center_x - radius // 3
        right_eye_x = center_x + radius // 3
        mouth_y = center_y + radius // 4

        # Eyebrows
        if expression == "angry":
            cv2.line(
                frame,
                (left_eye_x - 35, eye_y - 35),
                (left_eye_x + 25, eye_y - 15),
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.line(
                frame,
                (right_eye_x - 25, eye_y - 15),
                (right_eye_x + 35, eye_y - 35),
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )

        elif expression == "confused":
            cv2.line(
                frame,
                (left_eye_x - 30, eye_y - 30),
                (left_eye_x + 30, eye_y - 42),
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.line(
                frame,
                (right_eye_x - 30, eye_y - 42),
                (right_eye_x + 30, eye_y - 30),
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )

        # Eyes
        if expression == "sleepy":
            cv2.line(frame, (left_eye_x - 25, eye_y), (left_eye_x + 25, eye_y), (0, 0, 0), 5)
            cv2.line(frame, (right_eye_x - 25, eye_y), (right_eye_x + 25, eye_y), (0, 0, 0), 5)

            cv2.putText(
                frame,
                "Zzz",
                (center_x + radius - 25, center_y - radius + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                self.soft_white,
                3,
                cv2.LINE_AA,
            )

        elif expression == "crying":
            cv2.circle(frame, (left_eye_x, eye_y), radius // 10, (0, 0, 0), -1)
            cv2.circle(frame, (right_eye_x, eye_y), radius // 10, (0, 0, 0), -1)

            # Tears
            cv2.ellipse(frame, (left_eye_x, eye_y + 35), (11, 28), 0, 0, 360, (255, 170, 0), -1)
            cv2.ellipse(frame, (right_eye_x, eye_y + 35), (11, 28), 0, 0, 360, (255, 170, 0), -1)

        elif expression == "excited":
            self.draw_star(frame, left_eye_x, eye_y, radius // 8, self.soft_white)
            self.draw_star(frame, right_eye_x, eye_y, radius // 8, self.soft_white)

        else:
            cv2.circle(frame, (left_eye_x, eye_y), radius // 10, (0, 0, 0), -1)
            cv2.circle(frame, (right_eye_x, eye_y), radius // 10, (0, 0, 0), -1)
            cv2.circle(frame, (left_eye_x - 4, eye_y - 4), radius // 28, self.soft_white, -1)
            cv2.circle(frame, (right_eye_x - 4, eye_y - 4), radius // 28, self.soft_white, -1)

        # Mouth
        if expression == "happy":
            cv2.ellipse(
                frame,
                (center_x, mouth_y),
                (radius // 2, radius // 3),
                0,
                0,
                180,
                (0, 0, 0),
                6,
                cv2.LINE_AA,
            )

        elif expression == "excited":
            cv2.ellipse(
                frame,
                (center_x, mouth_y),
                (radius // 2, radius // 3),
                0,
                0,
                180,
                (0, 0, 0),
                7,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "!",
                (center_x + radius - 10, center_y - radius + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                self.soft_white,
                4,
                cv2.LINE_AA,
            )

        elif expression == "shocked":
            cv2.circle(frame, (center_x, mouth_y), radius // 4, (0, 0, 0), -1)

        elif expression == "sleepy":
            cv2.ellipse(frame, (center_x, mouth_y), (radius // 4, radius // 9), 0, 0, 360, (0, 0, 0), 3)

        elif expression == "sad":
            cv2.ellipse(
                frame,
                (center_x, mouth_y + radius // 4),
                (radius // 2, radius // 3),
                0,
                180,
                360,
                (0, 0, 0),
                6,
                cv2.LINE_AA,
            )

        elif expression == "crying":
            cv2.ellipse(
                frame,
                (center_x, mouth_y + radius // 5),
                (radius // 2, radius // 3),
                0,
                180,
                360,
                (0, 0, 0),
                6,
                cv2.LINE_AA,
            )

        elif expression == "angry":
            cv2.line(
                frame,
                (center_x - radius // 3, mouth_y + 20),
                (center_x + radius // 3, mouth_y),
                (0, 0, 0),
                6,
                cv2.LINE_AA,
            )

        elif expression == "neutral":
            cv2.line(
                frame,
                (center_x - radius // 3, mouth_y),
                (center_x + radius // 3, mouth_y),
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )

        else:
            cv2.line(
                frame,
                (center_x - radius // 3, mouth_y),
                (center_x + radius // 5, mouth_y + 18),
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )

        return frame

    def draw_npc_bubble(self, frame, x, y, box_w, box_h, expression):
        """
        Draw NPC speech bubble.
        """

        info = self.expression_info[expression]

        bubble_w = 500
        bubble_h = 145
        bubble_x = x + box_w + 28
        bubble_y = y + 10

        if bubble_x + bubble_w > frame.shape[1]:
            bubble_x = max(25, x - bubble_w - 28)

        bubble_y = max(125, min(bubble_y, frame.shape[0] - bubble_h - 25))

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (bubble_x + 10, bubble_y + 10),
            (bubble_x + bubble_w + 10, bubble_y + bubble_h + 10),
            (0, 0, 0),
            -1,
        )

        cv2.rectangle(
            overlay,
            (bubble_x, bubble_y),
            (bubble_x + bubble_w, bubble_y + bubble_h),
            self.card_dark,
            -1,
        )

        frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)

        frame = self.draw_glow_rect(
            frame,
            (bubble_x, bubble_y),
            (bubble_x + bubble_w, bubble_y + bubble_h),
            self.neon_magenta,
            2,
        )

        cv2.rectangle(
            frame,
            (bubble_x, bubble_y),
            (bubble_x + bubble_w, bubble_y + 38),
            (120, 40, 160),
            -1,
        )

        if bubble_x > x:
            pointer = np.array(
                [
                    [bubble_x, bubble_y + 60],
                    [bubble_x - 28, bubble_y + 75],
                    [bubble_x, bubble_y + 90],
                ]
            )
        else:
            pointer = np.array(
                [
                    [bubble_x + bubble_w, bubble_y + 60],
                    [bubble_x + bubble_w + 28, bubble_y + 75],
                    [bubble_x + bubble_w, bubble_y + 90],
                ]
            )

        cv2.fillPoly(frame, [pointer], self.card_dark)
        cv2.polylines(frame, [pointer], True, self.neon_magenta, 2)

        cv2.putText(
            frame,
            info["title"],
            (bubble_x + 16, bubble_y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            self.soft_white,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            info["message"],
            (bubble_x + 18, bubble_y + 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            self.soft_white,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            info["status"],
            (bubble_x + 18, bubble_y + 113),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            self.muted_white,
            1,
            cv2.LINE_AA,
        )

        return frame

    def draw_status_panel(self, frame, expression):
        """
        Draw bottom-left status panel.
        """

        h, w, _ = frame.shape
        info = self.expression_info[expression]

        panel_x = 25
        panel_y = h - 180
        panel_w = 470
        panel_h = 135

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (panel_x + 10, panel_y + 10),
            (panel_x + panel_w + 10, panel_y + panel_h + 10),
            (0, 0, 0),
            -1,
        )

        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            self.card_dark,
            -1,
        )

        frame = cv2.addWeighted(overlay, 0.86, frame, 0.14, 0)

        frame = self.draw_glow_rect(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            self.neon_cyan,
            2,
        )

        cv2.putText(
            frame,
            "LIVE NPC STATUS",
            (panel_x + 18, panel_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            self.neon_cyan,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Mood: {info['mood']}",
            (panel_x + 18, panel_y + 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            self.soft_white,
            2,
            cv2.LINE_AA,
        )

        meter_x = panel_x + 18
        meter_y = panel_y + 86
        meter_w = 400
        meter_h = 18

        cv2.rectangle(
            frame,
            (meter_x, meter_y),
            (meter_x + meter_w, meter_y + meter_h),
            (55, 55, 55),
            -1,
        )

        fill_w = int((info["meter"] / 100) * meter_w)

        if expression in ["happy", "excited"]:
            meter_color = self.neon_green
        elif expression in ["shocked", "angry", "crying"]:
            meter_color = self.neon_red
        elif expression == "sleepy":
            meter_color = self.neon_orange
        else:
            meter_color = self.neon_cyan

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (meter_x, meter_y),
            (meter_x + fill_w, meter_y + meter_h),
            meter_color,
            -1,
        )
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        cv2.rectangle(
            frame,
            (meter_x, meter_y),
            (meter_x + fill_w, meter_y + meter_h),
            meter_color,
            -1,
        )

        cv2.rectangle(
            frame,
            (meter_x, meter_y),
            (meter_x + meter_w, meter_y + meter_h),
            self.soft_white,
            1,
        )

        cv2.putText(
            frame,
            info["status"],
            (panel_x + 18, panel_y + 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            self.muted_white,
            1,
            cv2.LINE_AA,
        )

        return frame

    def draw_emotion_card(self, frame, expression):
        """
        Draw right-side detected emotion card.
        """

        h, w, _ = frame.shape

        card_w = 285
        card_h = 82
        x = w - card_w - 24
        y = 130

        mode_label = "AUTO" if not self.demo_override else "MANUAL"

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x + 8, y + 8),
            (x + card_w + 8, y + card_h + 8),
            (0, 0, 0),
            -1,
        )

        cv2.rectangle(
            overlay,
            (x, y),
            (x + card_w, y + card_h),
            self.card_dark,
            -1,
        )

        frame = cv2.addWeighted(overlay, 0.86, frame, 0.14, 0)

        frame = self.draw_glow_rect(
            frame,
            (x, y),
            (x + card_w, y + card_h),
            self.neon_purple,
            2,
        )

        cv2.putText(
            frame,
            f"EMOTION MODE: {mode_label}",
            (x + 16, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            self.neon_purple,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            expression.upper(),
            (x + 16, y + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            self.soft_white,
            2,
            cv2.LINE_AA,
        )

        return frame