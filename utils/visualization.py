import os
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
]


def get_color(class_id):
    return COLORS[class_id % len(COLORS)]


def draw_detections(image, detections, show_labels=True, show_confidence=True):
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        image = np.array(image)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    img = image.copy()
    
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = [int(x) for x in bbox]
        class_id = det.get("class_id", 0)
        class_name = det.get("class_name", str(class_id))
        confidence = det.get("confidence", 0.0)
        
        color = get_color(class_id)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        if show_labels or show_confidence:
            label = ""
            if show_labels:
                label += class_name
            if show_confidence:
                label += f" {confidence:.2f}"
            
            if label:
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    img,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
    
    return img


def draw_classification(image, results, top_k=3):
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        image = np.array(image)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    img = image.copy()
    h, w = img.shape[:2]
    
    bar_height = 30
    bar_width = int(w * 0.6)
    start_x = 20
    start_y = 40
    
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (start_x - 10, start_y - 30),
        (start_x + bar_width + 10, start_y + top_k * bar_height + 20),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    cv2.putText(
        img, "Classification Results:",
        (start_x, start_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    )
    
    for i, result in enumerate(results[:top_k]):
        class_name = result.get("class_name", str(result.get("class_id", i)))
        confidence = result.get("confidence", 0.0)
        
        y_pos = start_y + i * bar_height + 15
        
        cv2.putText(
            img,
            f"{i + 1}. {class_name}",
            (start_x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        
        bar_start_x = start_x + 180
        bar_end_x = bar_start_x + int((bar_width - 200) * confidence)
        cv2.rectangle(
            img,
            (bar_start_x, y_pos - 15),
            (bar_end_x, y_pos + 10),
            get_color(i),
            -1,
        )
        
        cv2.putText(
            img,
            f"{confidence:.1%}",
            (bar_start_x + 10, y_pos + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
    
    return img


def plot_training_curves(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history["train_losses"], label="Train Loss")
    ax1.plot(history["val_losses"], label="Val Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history["train_accs"], label="Train Acc")
    ax2.plot(history["val_accs"], label="Val Acc")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        return fig


def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Normalized Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8,
            )
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        return fig


def save_image(image, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if isinstance(image, np.ndarray):
        cv2.imwrite(output_path, image)
    elif isinstance(image, Image.Image):
        image.save(output_path)
    else:
        raise ValueError("Unsupported image type")
    
    return output_path
