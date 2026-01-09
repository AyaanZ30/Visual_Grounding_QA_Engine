import torch
from PIL import Image
import matplotlib.pyplot as plt

from models.DINO import GroundingDINO   
from utils.visualize import draw_boxes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    image_path =  "QA_sys.png"
    image = Image.open(image_path).convert("RGB")
    
    text_labels = [["diagram", "box"]]
    
    model = GroundingDINO(device = DEVICE)
    
    boxes, scores, labels = model.detect(image, text_labels)
    print("Detected labels : ",labels)
    print("Detection scores : ",scores)
    
    output = draw_boxes(image, boxes, labels)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(output)
    plt.axis("off")
    
    plt.savefig("/kaggle/working/output.png")
    print("Saved output.png")
    
    plt.show()

if __name__ == "__main__":
    main()