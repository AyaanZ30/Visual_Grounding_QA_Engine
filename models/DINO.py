import os
import torch 
from typing import List
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

class GroundingDINO:
    def __init__(self, device = "cpu", model_id = "IDEA-Research/grounding-dino-tiny"):
        self.device = device 
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
        ).to(device)
    
    def detect(self, image : Image, text_labels : list[list], box_threshold=0.35, text_threshold=0.25):
        inputs = self.processor(
            images = image,
            text = text_labels,
            return_tensors = "pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes = [image.size[::-1]]
        )[0]
        
        return result["boxes"], result["scores"], result["labels"]   
