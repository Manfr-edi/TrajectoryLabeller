import cv2
from pathlib import Path
from ultralytics import YOLO
import tkinter as tk
from tkinter.simpledialog import askinteger
from PIL import Image, ImageTk
import numpy as np

# --------------------------------
# CONFIG
# --------------------------------
CAM0_FOLDER = "./synchronized_frames/cam0"
CAM1_FOLDER = "./synchronized_frames/cam1"
MODEL_PATH = "yolov10l.pt"

# --------------------------------
# LOAD MODEL
# --------------------------------
model = YOLO(MODEL_PATH)

# --------------------------------
# LOAD FRAMES
# --------------------------------
cam0_frames = sorted(Path(CAM0_FOLDER).glob("*.jpeg"))
cam1_frames = sorted(Path(CAM1_FOLDER).glob("*.jpeg"))

num_frames = min(len(cam0_frames), len(cam1_frames))

# --------------------------------
# CACHE
# --------------------------------
detections_cache = {0: {}, 1: {}}
manual_overrides = {0: {}, 1: {}}

print("Running tracking...")

for i in range(num_frames):
    r0 = model.track(str(cam0_frames[i]), persist=True, tracker="botsort.yaml", verbose=False)
    r1 = model.track(str(cam1_frames[i]), persist=True, tracker="botsort.yaml", verbose=False)

    detections_cache[0][i] = r0
    detections_cache[1][i] = r1

print("Tracking done.")

# --------------------------------
# GUI
# --------------------------------
root = tk.Tk()
root.title("Dual Camera YOLO ID Editor")

canvas = tk.Label(root)
canvas.pack()

slider = tk.Scale(root, from_=0, to=num_frames-1, orient=tk.HORIZONTAL, length=1000)
slider.pack()

current_frame = 0

# --------------------------------
# DRAW
# --------------------------------
def draw_frame(index):

    global current_frame
    current_frame = index

    img0 = cv2.imread(str(cam0_frames[index]))
    img1 = cv2.imread(str(cam1_frames[index]))

    img0 = draw_boxes(img0, 0, index)
    img1 = draw_boxes(img1, 1, index)

    combined = np.hstack((img0, img1))

    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(combined_rgb)
    imgtk = ImageTk.PhotoImage(img_pil)

    canvas.imgtk = imgtk
    canvas.configure(image=imgtk)


def draw_boxes(img, cam_id, frame_index):

    results = detections_cache[cam_id][frame_index][0]
    boxes = results.boxes

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        track_id = int(box.id.item()) if box.id is not None else -1

        if frame_index in manual_overrides[cam_id]:
            if track_id in manual_overrides[cam_id][frame_index]:
                track_id = manual_overrides[cam_id][frame_index][track_id]

        x1, y1, x2, y2 = xyxy

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"ID:{track_id}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return img


# --------------------------------
# CLICK HANDLER
# --------------------------------
def on_click(event):

    frame_index = current_frame

    # Determine which camera was clicked
    img0 = cv2.imread(str(cam0_frames[frame_index]))
    h, w, _ = img0.shape

    if event.x < w:
        cam_id = 0
        click_x = event.x
    else:
        cam_id = 1
        click_x = event.x - w

    click_y = event.y

    results = detections_cache[cam_id][frame_index][0]
    boxes = results.boxes

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        track_id = int(box.id.item()) if box.id is not None else -1

        x1, y1, x2, y2 = xyxy

        if x1 <= click_x <= x2 and y1 <= click_y <= y2:

            new_id = askinteger("Change ID", f"Cam {cam_id}\nCurrent ID: {track_id}\nNew ID:")
            if new_id is not None:

                if frame_index not in manual_overrides[cam_id]:
                    manual_overrides[cam_id][frame_index] = {}

                manual_overrides[cam_id][frame_index][track_id] = new_id

                draw_frame(frame_index)
            break

canvas.bind("<Button-1>", on_click)

# --------------------------------
# SLIDER
# --------------------------------
def on_slider(val):
    draw_frame(int(val))

slider.config(command=on_slider)

draw_frame(0)

root.mainloop()
