import torch
from PIL import Image
import matplotlib.pyplot as plt

from models.DINO import GroundingDINO  
from models.BLIP import BLIPVisualReasoning 
from utils.visualize import draw_boxes
from utils.entity_extractor import extract_entities

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    image_path =  "/kaggle/input/abc/pytorch/default/1/few_random_imgs/people.jpg"
    image = Image.open(image_path).convert("RGB")
    
    # text_labels = [["cat", "man"]]
    question = "How many people are there?"
    
    vqa_model = BLIPVisualReasoning(device = DEVICE, use_fp16 = True)
    
    grounding_model = GroundingDINO(device = DEVICE)
    
    answer = vqa_model.answer(image, question)
    print("Answer : ",answer)
    
    entities =extract_entities(answer)
    
    boxes, scores, labels = grounding_model.detect(image, [[entities]])
    # boxes, scores, labels = grounding_model.detect(image, answer)
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