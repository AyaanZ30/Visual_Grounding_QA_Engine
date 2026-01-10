# extract_entities({BLIP_output}) -> [entity_1, ..]
# extract_entities("An umbrella") -> ["umbrella"](a text obj) -> DINO() -> BB (around umbrella)
from google import genai
from pydantic import BaseModel
from typing import List 

class EntityList(BaseModel):
    entities : List[str]

client = genai.Client(api_key = "AIzaSyCmFHIxMW-KtQ8xRQ8LmOlV3zwFUs8Qs0I")

def extract_entities(text : str) -> List[str]:
    if not text:
        return []
    
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = f"Extract all physical object (entities) from the text : {text}",
        config = {
            'response_mime_type' : 'application/json',
            'response_schema' : EntityList
        }
    )
    
    return response.parsed.entities

# ent = extract_entities("A person holding a yellow umbrella standing next to a blue car.")
# print(ent)
    
