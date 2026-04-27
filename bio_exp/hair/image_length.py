import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2
import os

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title='Select an image')
    return file_path

def load_new_image():
    global img, points, ref_length, selected_point, image_path
    new_image_path = select_image()
    if new_image_path:
        image_path = new_image_path
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        points.clear()
        ref_length = None
        selected_point = None
        redraw()

def save_figure():
    if image_path:
        os.makedirs("plots", exist_ok=True)
        image_name = os.path.basename(image_path)
        image_name_without_ext = os.path.splitext(image_name)[0]  # Remove file extension
        save_path = os.path.join("plots", f"plot_{image_name_without_ext}.png")
        
        fig.savefig(save_path) 

        print(f"Figure saved as {save_path}")

def redraw():
    ax.clear()
    ax.imshow(img)
    global ref_length
    
    if len(points) >= 2:
        ax.plot(*zip(*points[:2]), 'bo', markersize=8)
        ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], 'b-')
        ref_length = euclidean_distance(points[0], points[1])
    
    if len(points) == 4:
        ax.plot(*zip(*points[2:]), 'go', markersize=8)
        ax.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], 'g-')
        measured_length = euclidean_distance(points[2], points[3])
        relative_length = measured_length / ref_length if ref_length else float('inf')
        plt.title(f"Measured length: {relative_length:.3f}x of reference")
    else:
        plt.title("Click two points for reference, then two for measurement")
    
    fig.canvas.draw()

def onclick(event):
    global selected_point
    if event.xdata is not None and event.ydata is not None:
        for i, point in enumerate(points):
            if euclidean_distance((event.xdata, event.ydata), point) < 10:
                selected_point = i
                return
        if len(points) < 4:
            points.append((event.xdata, event.ydata))
            redraw()

def onmotion(event):
    if selected_point is not None and event.xdata is not None and event.ydata is not None:
        points[selected_point] = (event.xdata, event.ydata)
        redraw()

def onrelease(event):
    global selected_point
    selected_point = None

def on_key(event):
    if event.key == 'n':  # Press 'n' to load a new image
        load_new_image()
    elif event.key == 'd':  # Press 's' to save the figure
        save_figure()

image_path = select_image()
if image_path:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    points = []
    ref_length = None
    selected_point = None
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click two points for reference, then two for measurement")
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('motion_notify_event', onmotion)
    fig.canvas.mpl_connect('button_release_event', onrelease)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()
else:
    print("No image selected.")
