import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from matplotlib import colors as mcolors

# Setup Detectron2 configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Threshold for detection
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # COCO dataset classes
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

color_dict = {name: mcolors.to_rgb(name) for name in mcolors.CSS4_COLORS}

def closest_color(rgb_color):
    
    if sum((255 - c) for c in rgb_color) < 30:  
        return "white"

    
    min_distance = float('inf')
    closest_name = None
    for color_name, color_rgb in color_dict.items():
        color_rgb_255 = tuple(int(c * 255) for c in color_rgb)
        distance = sum((rgb_color[i] - color_rgb_255[i]) ** 2 for i in range(3))
        if distance < min_distance:
            min_distance = distance
            closest_name = color_name
    return closest_name



def get_dominant_color_kmeans(image, mask, n_clusters=3):
    
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked_image[mask == 255].reshape(-1, 3)
    
    # Use KMeans to find dominant color clusters
    if len(pixels) == 0:
        return (0, 0, 0)  # Return black if no valid pixels are found
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    cluster_centers = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
   
    most_abundant_color = cluster_centers[np.argmax(counts)]
    return tuple(most_abundant_color)

def segment_and_detect_colors(image_path):
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return
    image = cv2.resize(image, (640, 640))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Image loaded and resized.")

   
    outputs = predictor(image)
    print("Detection complete.")

    
    instances = outputs["instances"]
    car_instances = instances[instances.pred_classes == 2]

    if len(car_instances) == 0:
        print("No car detected.")
        return

    
    car_mask = car_instances.pred_masks[0].cpu().numpy().astype(np.uint8) * 255
    car_body_region = cv2.bitwise_and(image_rgb, image_rgb, mask=car_mask)
    print("Car mask applied, background removed.")

    
    h, w = car_body_region.shape[:2]
    upper_car_region = car_body_region[:int(h * 0.5), :]  # Upper half of the car
    grayscale_upper_car = cv2.cvtColor(upper_car_region, cv2.COLOR_RGB2GRAY)
    _, window_region_mask = cv2.threshold(grayscale_upper_car, 70, 255, cv2.THRESH_BINARY_INV)
    window_region_mask = np.pad(window_region_mask, ((0, h - window_region_mask.shape[0]), (0, w - window_region_mask.shape[1])), 'constant')
    window_region_mask = cv2.bitwise_and(car_mask, window_region_mask.astype(np.uint8))

    
    interior_rgb = get_dominant_color_kmeans(image_rgb, window_region_mask)
    
    
    exterior_mask = car_mask.copy()
    exterior_mask[window_region_mask == 255] = 0  

    
    exterior_rgb = get_dominant_color_kmeans(image_rgb, exterior_mask)
    
    
    exterior_color_name = closest_color(exterior_rgb)
    interior_color_name = closest_color(interior_rgb)

    
    print(f"Exterior color (Body): {exterior_color_name} (RGB{exterior_rgb})")
    print(f"Interior color (Window): {interior_color_name} (RGB{interior_rgb})")

  
    plt.figure(figsize=(10, 5))
    
   
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.bitwise_and(image_rgb, image_rgb, mask=exterior_mask))
    plt.title(f"Car Exterior: {exterior_color_name} (RGB{exterior_rgb})")
    plt.axis('off')
    
    
    plt.subplot(1, 2, 2)
    window_region = cv2.bitwise_and(image_rgb, image_rgb, mask=window_region_mask)
    plt.imshow(window_region)
    plt.title(f"Car Interior: {interior_color_name} (RGB{interior_rgb})")
    plt.axis('off')
    
    plt.show()

# Path to your car image
image_path = r"C:\Users\ashut\OneDrive\Pictures\pyhton projects\yellow lambo.jpg"
segment_and_detect_colors(image_path)


