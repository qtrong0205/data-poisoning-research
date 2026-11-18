import numpy as np

def flip_labels_random(labels, rate, num_classes, seed):
    """Đổi nhãn ngẫu nhiên một tỷ lệ mẫu sang một lớp KHÁC nhãn gốc."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    new = labels.copy()
    
    n = int(round(rate * len(labels)))
    if n == 0:
        return new, np.array([])
        
    idx = rng.choice(len(labels), size=n, replace=False)
    shift = rng.integers(1, num_classes, size=n)      
    new[idx] = (labels[idx] + shift) % num_classes
    
    return new, np.sort(idx)

def flip_labels_targeted(labels, rate, source_label, target_label, seed, denominator="source_class"):
    """Đổi nhãn một phần mẫu của source_label sang target_label."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    new = labels.copy()
    
    src = np.where(labels == source_label)[0]
    if len(src) == 0:
        return new, np.array([])
        
    n = round(rate * len(src)) if denominator == "source_class" \
        else min(round(rate * len(labels)), len(src))
        
    if n == 0:
        return new, np.array([])
    
    idx = rng.choice(src, size=int(n), replace=False)
    new[idx] = target_label
    
    return new, np.sort(idx)