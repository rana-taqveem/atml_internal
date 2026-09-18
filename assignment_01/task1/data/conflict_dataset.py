import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class ConflictDataset(Dataset):
    
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.records = load_conflict_records(self.dataset_dir)
        self.to_tensor = ToTensor()
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, index):
        record = self.records[index]
        image_path = self.dataset_dir / record["image_path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if image.size != (224, 224):
                raise ValueError(f"Expected a 224×224 image: {image_path}")

            image_tensor = self.to_tensor(image)

        return {
            "image": image_tensor,
            "conflict_id": record["conflict_id"],
            "content_id": record["content_id"],
            "style_id": record["style_id"],
            "content_label": record["content_label"],
            "style_label": record["style_label"],
        }
    
    
def load_conflict_records(dataset_dir):
    
    dataset_dir = Path(dataset_dir)
    
    with (dataset_dir / "manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    records = []
    seen_ids = set()
    
    for entry in manifest:
        metadata_path = dataset_dir / entry["metadata_path"]
        with metadata_path.open("r", encoding="utf-8") as f:
            record = json.load(f)
            
        conflict_id = record["conflict_id"]
        
        if conflict_id != entry["conflict_id"]:
            raise ValueError(f"Conflict ID {conflict_id} found in content or style IDs.")
        
        if conflict_id in seen_ids:
            raise ValueError(f"Duplicate conflict ID found: {conflict_id}")
        
        if record["accepted"] is not True or record["selected_for_evaluation"] is not True:
            raise ValueError(f"Conflict ID {conflict_id} is not accepted or not selected for evaluation.")
        
        if record["content_label"] == record["style_label"]:
            raise ValueError(f"Conflict ID {conflict_id} has identical content and style labels.")
        
        image_path = dataset_dir / record["image_path"] 
        
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found for conflict ID {conflict_id}: {image_path}")
        
        seen_ids.add(conflict_id)
        records.append(record)
        
    if not records:
        raise ValueError("No valid conflict records found in the dataset.")
    
    return records

        
        