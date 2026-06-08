import cv2
import numpy as np
import mediapipe as mp


class FunnyFaceEmojiTask:
    """
    Special Task:
    Expression-Based NPC Emoji Cam.

    Uses MediaPipe Face Mesh, a pretrained neural-network landmark model,
    to detect facial landmarks in real time.

    Then it uses simple landmark-distance rules to estimate expressions:
    happy, shocked, sleepy, sad, angry, crying, excited, neutral/confused.

    Finally, OpenCV draws:
    - Emoji face overlay
    - NPC-style speech bubble
    - Game-style frontend HUD
    - Mood/status panel
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.show_landmarks = False
        self.last_expression = "neutral"

        self.expression_messages = {
            "happy": {
                "title": "Happy NPC Student",
                "message": "Finally understood the lecture!",
                "status": "Confidence: 91% | Confusion: 10%",
                "mood": "GOOD MOOD",
                "meter": 85
            },
            "excited": {
                "title": "Main Character Mode",
                "message": "Project demo is looking amazing!",
                "status": "Energy: 96% | Confidence: 94%",
                "mood": "EXCITED",
                "meter": 96
            },
            "shocked": {
                "title": "Deadline Boss Appeared",
                "message": "Assignment deadline detected!",
                "status": "Panic: 99% | Sleep: 3%",
                "mood": "PANIC",
                "meter": 99
            },
            "sleepy": {
                "title": "Sleepy Student",
                "message": "Energy low. Coffee required.",
                "status": "Energy: 12% | Focus: 21%",
                "mood": "SLEEPY",
                "meter": 18
            },
            "sad": {
                "title": "Sad NPC Student",
                "message": "Code compiled but the grade is unknown.",
                "status": "Motivation: 34% | Stress: 72%",
                "mood": "SAD",
                "meter": 35
            },
            "crying": {
                "title": "Crying Debugger",
                "message": "One error fixed, three new errors appeared.",
                "status": "Tears: 88% | Debugging: 99%",
                "mood": "CRYING",
                "meter": 25
            },
            "angry": {
                "title": "Angry Compiler Victim",
                "message": "Why is this import not working?!",
                "status": "Anger: 84% | Patience: 9%",
                "mood": "ANGRY",
                "meter": 20
            },
            "neutral": {
                "title": "NPC Student Lv. 3",
                "message": "Processing lecture...",
                "status": "Focus: 50% | Confusion: 50%",
                "mood": "NEUTRAL",
                "meter": 50
            },
            "confused": {
                "title": "Confused NPC Detected",
                "message": "Quest: Survive Computer Vision.",
                "status": "Confusion: 88% | Deadline Fear: 76%",
                "mood": "CONFUSED",
                "meter": 42
            }
        }

    def set_mode_from_key(self, key):
        """
        Keyboard control.
        Press l to show/hide landmarks.
        """
        if key == ord("l"):
            self.show_landmarks = not self.show_landmarks

    def process_frame(self, frame):
        """
        Main function called from main.py.
        """

        output = frame.copy()
        h, w, _ = output.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        output = self.draw_frontend(output)

        if not results.multi_face_landmarks:
            output = self.draw_no_face_screen(output)
            return output

        face_landmarks = results.multi_face_landmarks[0]

        points = []
        for lm in face_landmarks.landmark:
            points.append((int(lm.x * w), int(lm.y * h)))

        x, y, box_w, box_h = self.get_face_box(points, w, h)

        expression = self.detect_expression(points)
        self.last_expression = expression

        if self.show_landmarks:
            output = self.draw_landmarks(output, points)

        output = self.draw_emoji_overlay(output, x, y, box_w, box_h, expression)
        output = self.draw_npc_bubble(output, x, y, box_w, box_h, expression)
        output = self.draw_status_panel(output, expression)
        output = self.draw_expression_badge(output, expression)

        return output

    def get_face_box(self, points, frame_w, frame_h):
        """
        Creates bounding box around the face.
        """

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min = max(0, min(xs) - 45)
        y_min = max(0, min(ys) - 65)
        x_max = min(frame_w, max(xs) + 45)
        y_max = min(frame_h, max(ys) + 55)

        return x_min, y_min, x_max - x_min, y_max - y_min

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def detect_expression(self, points):
        """
        Expression detection using Face Mesh landmark ratios.

        Note:
        Face Mesh is excellent at detecting face geometry.
        It does not directly give emotions.
        So these emotions are estimated using simple facial rules.

        More reliable:
        - shocked: mouth open
        - sleepy: eyes nearly closed
        - happy/excited: mouth shape/opening

        Less reliable but still demo-friendly:
        - sad
        - angry
        - crying
        """

        # Mouth
        left_mouth = points[61]
        right_mouth = points[291]
        upper_lip = points[13]
        lower_lip = points[14]

        # Face reference points
        nose_tip = points[1]
        chin = points[152]
        forehead = points[10]

        # Eyes
        left_eye_top = points[159]
        left_eye_bottom = points[145]
        right_eye_top = points[386]
        right_eye_bottom = points[374]

        # Eyebrows
        left_brow = points[105]
        right_brow = points[334]
        left_eye_center = points[159]
        right_eye_center = points[386]

        # Mouth corners
        left_corner = points[61]
        right_corner = points[291]
        mouth_center = (
            int((upper_lip[0] + lower_lip[0]) / 2),
            int((upper_lip[1] + lower_lip[1]) / 2)
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

        # If corners are visually lower than center, it looks sad.
        mouth_curve = (mouth_corner_avg_y - mouth_center_y) / mouth_width

        # Rules
        if eye_open_ratio < 0.045:
            return "sleepy"

        if mouth_open_ratio > 0.30 and eye_open_ratio > 0.075:
            return "shocked"

        if mouth_open_ratio > 0.22:
            return "excited"

        if mouth_open_ratio > 0.11:
            return "happy"

        if brow_distance < 0.045 and eye_open_ratio > 0.045:
            return "angry"

        if mouth_curve > 0.055:
            return "sad"

        if mouth_curve > 0.035 and eye_open_ratio < 0.060:
            return "crying"

        if 0.060 <= mouth_open_ratio <= 0.10:
            return "neutral"

        return "confused"

    def draw_frontend(self, frame):
        """
        Top modern HUD.
        """

        h, w, _ = frame.shape

        overlay = frame.copy()

        # Top glass panel
        cv2.rectangle(overlay, (0, 0), (w, 115), (10, 10, 25), -1)
        frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

        # Accent line
        cv2.line(frame, (0, 115), (w, 115), (0, 255, 255), 2)

        cv2.putText(
            frame,
            "EXPRESSION-BASED NPC EMOJI CAM",
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "MediaPipe Face Mesh Neural Network  +  OpenCV Real-Time Overlay",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Controls: f = Special Mode | l = Landmarks | t/e/s = Basic Modes | q = Quit",
            (25, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        return frame

    def draw_no_face_screen(self, frame):
        """
        UI when no face is detected.
        """

        h, w, _ = frame.shape

        cv2.rectangle(frame, (40, h - 120), (620, h - 45), (20, 20, 20), -1)
        cv2.rectangle(frame, (40, h - 120), (620, h - 45), (0, 255, 255), 2)

        cv2.putText(
            frame,
            "No student detected...",
            (60, h - 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Probably making coffee or avoiding the deadline.",
            (60, h - 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return frame

    def draw_landmarks(self, frame, points):
        """
        Draws small face mesh dots.
        """

        for p in points[::4]:
            cv2.circle(frame, p, 1, (0, 255, 0), -1)

        return frame

    def draw_emoji_overlay(self, frame, x, y, box_w, box_h, expression):
        """
        Draws better custom emoji over the real face.
        """

        center_x = x + box_w // 2
        center_y = y + box_h // 2
        radius = int(max(box_w, box_h) * 0.46)

        face_colors = {
            "happy": (0, 225, 255),
            "excited": (0, 240, 255),
            "shocked": (0, 205, 255),
            "sleepy": (210, 220, 255),
            "sad": (230, 190, 80),
            "crying": (240, 200, 90),
            "angry": (40, 80, 255),
            "neutral": (0, 210, 230),
            "confused": (80, 210, 255)
        }

        face_color = face_colors.get(expression, (0, 220, 255))

        # Shadow
        cv2.circle(frame, (center_x + 8, center_y + 8), radius, (20, 20, 20), -1)

        # Main face
        cv2.circle(frame, (center_x, center_y), radius, face_color, -1)
        cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 3)

        # Soft highlight
        cv2.circle(
            frame,
            (center_x - radius // 3, center_y - radius // 3),
            radius // 5,
            (255, 255, 255),
            -1
        )
        frame = cv2.addWeighted(frame, 0.96, frame, 0.04, 0)

        eye_y = center_y - radius // 4
        left_eye_x = center_x - radius // 3
        right_eye_x = center_x + radius // 3
        mouth_y = center_y + radius // 4

        # Draw eyebrows first
        if expression == "angry":
            cv2.line(
                frame,
                (left_eye_x - 35, eye_y - 35),
                (left_eye_x + 25, eye_y - 15),
                (0, 0, 0),
                5,
                cv2.LINE_AA
            )
            cv2.line(
                frame,
                (right_eye_x - 25, eye_y - 15),
                (right_eye_x + 35, eye_y - 35),
                (0, 0, 0),
                5,
                cv2.LINE_AA
            )

        elif expression == "confused":
            cv2.line(
                frame,
                (left_eye_x - 30, eye_y - 30),
                (left_eye_x + 30, eye_y - 42),
                (0, 0, 0),
                4,
                cv2.LINE_AA
            )
            cv2.line(
                frame,
                (right_eye_x - 30, eye_y - 42),
                (right_eye_x + 30, eye_y - 30),
                (0, 0, 0),
                4,
                cv2.LINE_AA
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
                1.1,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

        elif expression == "crying":
            cv2.circle(frame, (left_eye_x, eye_y), radius // 10, (0, 0, 0), -1)
            cv2.circle(frame, (right_eye_x, eye_y), radius // 10, (0, 0, 0), -1)

            # Tears
            cv2.ellipse(frame, (left_eye_x, eye_y + 35), (12, 28), 0, 0, 360, (255, 180, 40), -1)
            cv2.ellipse(frame, (right_eye_x, eye_y + 35), (12, 28), 0, 0, 360, (255, 180, 40), -1)

        elif expression == "excited":
            # Star eyes
            self.draw_star(frame, left_eye_x, eye_y, radius // 8, (255, 255, 255))
            self.draw_star(frame, right_eye_x, eye_y, radius // 8, (255, 255, 255))

        else:
            cv2.circle(frame, (left_eye_x, eye_y), radius // 10, (0, 0, 0), -1)
            cv2.circle(frame, (right_eye_x, eye_y), radius // 10, (0, 0, 0), -1)

            cv2.circle(frame, (left_eye_x - 4, eye_y - 4), radius // 28, (255, 255, 255), -1)
            cv2.circle(frame, (right_eye_x - 4, eye_y - 4), radius // 28, (255, 255, 255), -1)

        # Mouths
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
                cv2.LINE_AA
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
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                "!",
                (center_x + radius - 10, center_y - radius + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                4,
                cv2.LINE_AA
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
                cv2.LINE_AA
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
                cv2.LINE_AA
            )

        elif expression == "angry":
            cv2.line(
                frame,
                (center_x - radius // 3, mouth_y + 20),
                (center_x + radius // 3, mouth_y),
                (0, 0, 0),
                6,
                cv2.LINE_AA
            )

        elif expression == "neutral":
            cv2.line(
                frame,
                (center_x - radius // 3, mouth_y),
                (center_x + radius // 3, mouth_y),
                (0, 0, 0),
                5,
                cv2.LINE_AA
            )

        else:
            cv2.line(
                frame,
                (center_x - radius // 3, mouth_y),
                (center_x + radius // 5, mouth_y + 18),
                (0, 0, 0),
                5,
                cv2.LINE_AA
            )

        return frame

    def draw_star(self, frame, cx, cy, size, color):
        """
        Draw simple star for excited eyes.
        """

        points = np.array([
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
        ], np.int32)

        cv2.fillPoly(frame, [points], color)
        cv2.polylines(frame, [points], True, (0, 0, 0), 2)

    def draw_npc_bubble(self, frame, x, y, box_w, box_h, expression):
        """
        Draws NPC speech bubble.
        """

        info = self.expression_messages[expression]

        bubble_w = 480
        bubble_h = 135

        bubble_x = x + box_w + 25
        bubble_y = y + 15

        if bubble_x + bubble_w > frame.shape[1]:
            bubble_x = max(25, x - bubble_w - 25)

        bubble_y = max(130, min(bubble_y, frame.shape[0] - bubble_h - 30))

        overlay = frame.copy()

        # Bubble shadow
        cv2.rectangle(
            overlay,
            (bubble_x + 8, bubble_y + 8),
            (bubble_x + bubble_w + 8, bubble_y + bubble_h + 8),
            (0, 0, 0),
            -1
        )

        # Bubble main
        cv2.rectangle(
            overlay,
            (bubble_x, bubble_y),
            (bubble_x + bubble_w, bubble_y + bubble_h),
            (25, 25, 40),
            -1
        )

        frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)

        # Border
        cv2.rectangle(
            frame,
            (bubble_x, bubble_y),
            (bubble_x + bubble_w, bubble_y + bubble_h),
            (0, 255, 255),
            2
        )

        # Header strip
        cv2.rectangle(
            frame,
            (bubble_x, bubble_y),
            (bubble_x + bubble_w, bubble_y + 38),
            (0, 120, 180),
            -1
        )

        # Pointer
        if bubble_x > x:
            pointer = np.array([
                [bubble_x, bubble_y + 60],
                [bubble_x - 28, bubble_y + 75],
                [bubble_x, bubble_y + 90]
            ])
        else:
            pointer = np.array([
                [bubble_x + bubble_w, bubble_y + 60],
                [bubble_x + bubble_w + 28, bubble_y + 75],
                [bubble_x + bubble_w, bubble_y + 90]
            ])

        cv2.fillPoly(frame, [pointer], (25, 25, 40))
        cv2.polylines(frame, [pointer], True, (0, 255, 255), 2)

        cv2.putText(
            frame,
            info["title"],
            (bubble_x + 16, bubble_y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            info["message"],
            (bubble_x + 18, bubble_y + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            info["status"],
            (bubble_x + 18, bubble_y + 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1,
            cv2.LINE_AA
        )

        return frame

    def draw_status_panel(self, frame, expression):
        """
        Bottom-left status dashboard.
        """

        h, w, _ = frame.shape
        info = self.expression_messages[expression]

        panel_x = 25
        panel_y = h - 170
        panel_w = 455
        panel_h = 130

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (panel_x + 8, panel_y + 8),
            (panel_x + panel_w + 8, panel_y + panel_h + 8),
            (0, 0, 0),
            -1
        )

        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (18, 18, 35),
            -1
        )

        frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "LIVE NPC STUDENT STATUS",
            (panel_x + 18, panel_y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Mood: {info['mood']}",
            (panel_x + 18, panel_y + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Mood meter
        meter_x = panel_x + 18
        meter_y = panel_y + 88
        meter_w = 390
        meter_h = 20

        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (60, 60, 60), -1)

        fill_w = int((info["meter"] / 100) * meter_w)

        if expression in ["happy", "excited"]:
            meter_color = (0, 255, 0)
        elif expression in ["shocked", "angry", "crying"]:
            meter_color = (0, 0, 255)
        elif expression == "sleepy":
            meter_color = (255, 180, 0)
        else:
            meter_color = (0, 255, 255)

        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + fill_w, meter_y + meter_h), meter_color, -1)
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (255, 255, 255), 1)

        cv2.putText(
            frame,
            info["status"],
            (panel_x + 18, panel_y + 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        return frame

    def draw_expression_badge(self, frame, expression):
        """
        Floating badge on right side.
        """

        h, w, _ = frame.shape
        info = self.expression_messages[expression]

        badge_w = 260
        badge_h = 65

        x = w - badge_w - 25
        y = 135

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + badge_w, y + badge_h), (15, 15, 30), -1)
        frame = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)

        cv2.rectangle(frame, (x, y), (x + badge_w, y + badge_h), (255, 255, 255), 2)

        cv2.putText(
            frame,
            "DETECTED EMOTION",
            (x + 18, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            expression.upper(),
            (x + 18, y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame