import cv2, torch
import time, os
import mediapipe as mp
import numpy as np
from model import Model

current_path = os.getcwd()

cam_number = 0
flip = True
min_conf = 0.75
max_hands = 2
model_path = os.path.join(current_path, 'models/120.pt')

pen_color = (255, 0, 0)
eraser_size = 80
pen_size = 10


intermediate_step_gap = 4

cv2.namedWindow('Virtual Drawing Board', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Virtual Drawing Board', 1200, 800)

cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Control Panel', 400, 600)

canvas = np.ones((800, 1200, 3), dtype=np.uint8) * 255 

camera_width, camera_height = 320, 240
camera_pos = (1200 - camera_width - 20, 20)  


control_img = np.zeros((600, 400, 3), np.uint8)
control_img[:] = (240, 240, 240)  

def nothing(x):
    pass

cv2.createTrackbar('Red', 'Control Panel', 0, 255, nothing)
cv2.createTrackbar('Green', 'Control Panel', 0, 255, nothing)
cv2.createTrackbar('Blue', 'Control Panel', 0, 255, nothing)
cv2.createTrackbar('Pen Thickness', 'Control Panel', 5, 30, nothing)
cv2.createTrackbar('Line Smoothness', 'Control Panel', 10, 29, nothing)

color_preview = np.zeros((100, 350, 3), np.uint8)
color_preview_pos = (25, 180)

buttons = {
    'clear': {'pos': (25, 300), 'size': (350, 50), 'label': 'Clear Canvas', 'color': (200, 200, 200)},
    'save': {'pos': (25, 370), 'size': (350, 50), 'label': 'Save Drawing', 'color': (200, 200, 200)},
    'exit': {'pos': (25, 440), 'size': (350, 50), 'label': 'Exit Program', 'color': (200, 200, 200)}
}

def process_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:

        for btn_name, btn in buttons.items():
            bx, by = btn['pos']
            bw, bh = btn['size']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if btn_name == 'clear':
                    global circles
                    circles = []
                elif btn_name == 'save':
                    cv2.imwrite(f'Drawing_{int(time.time())}.png', canvas)
                    btn['color'] = (0, 255, 0)  
                elif btn_name == 'exit':
                    cv2.destroyAllWindows()
                    exit()

cv2.setMouseCallback('Control Panel', process_click)

cap = cv2.VideoCapture(cam_number)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=max_hands,
    min_detection_confidence=min_conf,
    min_tracking_confidence=min_conf
)
mp_draw = mp.solutions.drawing_utils

_lm_list = [
    mpHands.HandLandmark.WRIST,
    mpHands.HandLandmark.THUMB_CMC,
    mpHands.HandLandmark.THUMB_MCP,
    mpHands.HandLandmark.THUMB_IP,
    mpHands.HandLandmark.THUMB_TIP,
    mpHands.HandLandmark.INDEX_FINGER_MCP,
    mpHands.HandLandmark.INDEX_FINGER_DIP,
    mpHands.HandLandmark.INDEX_FINGER_PIP,
    mpHands.HandLandmark.INDEX_FINGER_TIP,
    mpHands.HandLandmark.MIDDLE_FINGER_MCP,
    mpHands.HandLandmark.MIDDLE_FINGER_DIP,
    mpHands.HandLandmark.MIDDLE_FINGER_PIP,
    mpHands.HandLandmark.MIDDLE_FINGER_TIP,
    mpHands.HandLandmark.RING_FINGER_MCP,
    mpHands.HandLandmark.RING_FINGER_DIP,
    mpHands.HandLandmark.RING_FINGER_PIP,
    mpHands.HandLandmark.RING_FINGER_TIP,
    mpHands.HandLandmark.PINKY_MCP,
    mpHands.HandLandmark.PINKY_DIP,
    mpHands.HandLandmark.PINKY_PIP,
    mpHands.HandLandmark.PINKY_TIP
]

def landmark_extract(hand_lms, mpHands):
    output_lms = []
    for lm in _lm_list:
        lms = hand_lms.landmark[lm]
        output_lms.append(lms.x)
        output_lms.append(lms.y)
        output_lms.append(lms.z)
    return output_lms

def is_position_out_of_bounds(position, top_left, bottom_right):
    return (
        position[0] > top_left[0] and position[0] < bottom_right[0]
        and position[1] > top_left[1] and position[1] < bottom_right[1]
    )

model = Model()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

action_map = {0: 'Draw', 1: 'Erase', 2: 'None'}


font = cv2.FONT_HERSHEY_SIMPLEX
font_small = cv2.FONT_HERSHEY_DUPLEX
fontColor = (30, 30, 30)  
lineType = 2

circles = []
was_drawing_last_frame = False
ptime = 0
ctime = 0


def update_ui():
    
    control_img[:] = (240, 240, 240)  
    
    cv2.putText(control_img, 'VIRTUAL DRAWING BOARD', (25, 40), 
                font, 0.8, (0, 100, 200), 2)
    cv2.putText(control_img, 'Control Panel', (25, 80), 
                font, 0.6, (50, 50, 50), 1)
    
    cv2.putText(control_img, 'Color Controls:', (25, 120), 
                font_small, 0.5, (50, 50, 50), 1)
    cv2.putText(control_img, 'Red', (25, 150), font_small, 0.4, (0, 0, 200), 1)
    cv2.putText(control_img, 'Green', (150, 150), font_small, 0.4, (0, 200, 0), 1)
    cv2.putText(control_img, 'Blue', (275, 150), font_small, 0.4, (200, 0, 0), 1)
    
    b = cv2.getTrackbarPos('Blue', 'Control Panel')
    g = cv2.getTrackbarPos('Green', 'Control Panel')
    r = cv2.getTrackbarPos('Red', 'Control Panel')
    color_preview[:] = (b, g, r)
    control_img[color_preview_pos[1]:color_preview_pos[1]+100, 
                color_preview_pos[0]:color_preview_pos[0]+350] = color_preview
    
    for btn_name, btn in buttons.items():
        bx, by = btn['pos']
        bw, bh = btn['size']

        cv2.rectangle(control_img, (bx, by), (bx+bw, by+bh), btn['color'], -1)


        cv2.rectangle(control_img, (bx, by), (bx+bw, by+bh), (100, 100, 100), 2)
        text_size = cv2.getTextSize(btn['label'], font_small, 0.6, 2)[0]
        text_x = bx + (bw - text_size[0]) // 2
        text_y = by + (bh + text_size[1]) // 2
        cv2.putText(control_img, btn['label'], (text_x, text_y), 
                    font_small, 0.6, (50, 50, 50), 2)
    
    # cv2.putText(control_img, f'FPS: {fps}', (25, 520), 
    #             font_small, 0.5, (50, 50, 50), 1)
    cv2.putText(control_img, f'Mode: {current_action}', (25, 550), 
                font_small, 0.5, (50, 50, 50), 1)
    cv2.putText(control_img, 'Press Q to quit', (25, 580), 
                font_small, 0.5, (50, 50, 50), 1)

current_action = "None"
while True:
    success, frame = cap.read()
    if not success:
        break
        
    if flip:
        frame = cv2.flip(frame, 1)
    
    frame = cv2.resize(frame, (camera_width, camera_height))
    camera_display = frame.copy()
    
    h, w, c = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    b = cv2.getTrackbarPos('Blue', 'Control Panel')
    g = cv2.getTrackbarPos('Green', 'Control Panel')
    r = cv2.getTrackbarPos('Red', 'Control Panel')
    t = cv2.getTrackbarPos('Pen Thickness', 'Control Panel')
    imd_step_gap = (cv2.getTrackbarPos('Line Smoothness', 'Control Panel') + 1) / 10
    
    intermediate_step_gap = imd_step_gap
    
    if not results.multi_hand_landmarks:
        was_drawing_last_frame = False
        cv2.putText(camera_display, 'No hand detected', (10, 30), 
                    font_small, 0.5, (0, 0, 255), 1)
        current_action = "None"
    else:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(camera_display, hand_landmarks, mpHands.HAND_CONNECTIONS)
            
            landmark_list = landmark_extract(hand_landmarks, mpHands)
            model_input = torch.tensor(landmark_list, dtype=torch.float).unsqueeze(0)
            action = action_map[torch.argmax(model.forward(model_input)).item()]
            current_action = action
            
            cv2.putText(camera_display, f'Mode: {action}', (10, camera_height - 10), 
                        font_small, 0.5, (255, 255, 255), 2)
            cv2.putText(camera_display, f'Mode: {action}', (10, camera_height - 10), 
                        font_small, 0.5, (0, 100, 200), 1)
            
            if action == 'Draw':
                pen_color = (b, g, r)
                pen_size = t
                index_x = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x
                index_y = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y
                pos_x = int(index_x * 1200)
                pos_y = int(index_y * 800)
                pos = (pos_x, pos_y)
                
                cv2.circle(canvas, pos, pen_size + 2, (200, 200, 200), 1)
                cv2.circle(canvas, pos, pen_size, pen_color, -1)
                
                cam_pos = (int(index_x * w), int(index_y * h))
                cv2.circle(camera_display, cam_pos, 10, (0, 255, 0), 2)
                
                if was_drawing_last_frame and len(circles) > 0:
                    prev_pos = circles[-1][0]
                    x_distance = pos[0] - prev_pos[0]
                    y_distance = pos[1] - prev_pos[1]
                    distance = (x_distance ** 2 + y_distance ** 2) ** 0.5
                    num_step_points = int(int(distance) // intermediate_step_gap) - 1
                    if num_step_points > 0:
                        x_normalized = x_distance / distance
                        y_normalized = y_distance / distance
                        for i in range(1, num_step_points + 1):
                            step_pos_x = prev_pos[0] + int(x_normalized * i)
                            step_pos_y = prev_pos[1] + int(y_normalized * i)
                            step_pos = (step_pos_x, step_pos_y)
                            circles.append((step_pos, pen_color, pen_size))
                
                circles.append((pos, pen_color, pen_size))
                was_drawing_last_frame = True
            else:
                was_drawing_last_frame = False
            
            if action == 'Erase':
                eraser_mid_x = int(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].x * 1200)
                eraser_mid_y = int(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].y * 800)
                eraser_mid = (eraser_mid_x, eraser_mid_y)
                
                bottom_right = (eraser_mid[0] + eraser_size, eraser_mid[1] + eraser_size)
                top_left = (eraser_mid[0] - eraser_size, eraser_mid[1] - eraser_size)
                
                cv2.rectangle(canvas, top_left, bottom_right, (255, 150, 150), 2)
                
                circles = [
                    (position, color, size)
                    for position, color, size in circles
                    if not is_position_out_of_bounds(position, top_left, bottom_right)
                ]
    
    canvas_clean = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    for position, color, size in circles:
        cv2.circle(canvas_clean, position, size, color, -1)
    
    canvas_with_camera = canvas_clean.copy()
    border_size = 5
    border_color = (100, 100, 100)
    cv2.rectangle(canvas_with_camera, 
                  (camera_pos[0]-border_size, camera_pos[1]-border_size),
                  (camera_pos[0]+camera_width+border_size, camera_pos[1]+camera_height+border_size),
                  border_color, border_size)
    canvas_with_camera[camera_pos[1]:camera_pos[1]+camera_height, 
                       camera_pos[0]:camera_pos[0]+camera_width] = camera_display
    
    cv2.putText(canvas_with_camera, 'VIRTUAL DRAWING BOARD', (30, 40), 
                font, 1.2, (0, 100, 200), 3)
    cv2.putText(canvas_with_camera, 'Draw in air with your finger tips', (30, 80), 
                font_small, 0.7, (50, 50, 50), 1)
    
    # ctime = time.time()
    # fps = round(1 / (ctime - ptime), 2) if (ctime - ptime) > 0 else 0
    # ptime = ctime
    # # 
    update_ui()
    
    cv2.imshow('Virtual Drawing Board', canvas_with_camera)
    cv2.imshow('Control Panel', control_img)
    
    for btn in buttons.values():
        btn['color'] = (200, 200, 200)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()