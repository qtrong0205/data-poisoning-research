import numpy as np
import json

class LabelFlipper:
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def random_flip(self, labels: list | np.ndarray, flip_rate: float = 0.1, random_seed: int = None) -> tuple:
        if random_seed is not None:
            np.random.seed(random_seed)
        
        labels_arr = np.array(labels).copy()
        num_samples = len(labels_arr)
        num_flip = int(flip_rate * num_samples)
        
        flip_indices = np.random.choice(num_samples, num_flip, replace=False)
        manifest = {}
        
        for idx in flip_indices:
            orig_label = int(labels_arr[idx])
            possible_labels = [i for i in range(self.num_classes) if i != orig_label]
            new_label = np.random.choice(possible_labels)
            
            labels_arr[idx] = new_label
            manifest[int(idx)] = {
                "original_label": orig_label, 
                "poisoned_label": int(new_label)
            }
            
        return labels_arr, manifest

    def targeted_flip(self, labels: list | np.ndarray, source_class: int, target_class: int, flip_rate: float = 0.5, random_seed: int = None) -> tuple:
        if random_seed is not None:
            np.random.seed(random_seed)
            
        labels_arr = np.array(labels).copy()
        source_indices = np.where(labels_arr == source_class)[0]
        num_flip = int(flip_rate * len(source_indices))
        
        manifest = {}
        if num_flip == 0:
            return labels_arr, manifest
            
        flip_indices = np.random.choice(source_indices, num_flip, replace=False)
        
        for idx in flip_indices:
            orig_label = int(labels_arr[idx])
            labels_arr[idx] = target_class
            manifest[int(idx)] = {
                "original_label": orig_label, 
                "poisoned_label": int(target_class)
            }
            
        return labels_arr, manifest

    def save_manifest(self, manifest: dict, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=4)