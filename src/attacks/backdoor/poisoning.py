from pathlib import Path
from typing import Mapping

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALID_POSITIONS = {"bottom_right", "top_left", "center"}


def create_trigger(size_px, color, image_mode):
    if not isinstance(size_px, int) or isinstance(size_px, bool) or size_px <= 0:
        raise ValueError("Trigger size must be a positive integer.")

    return Image.new(mode=image_mode, size=(size_px, size_px), color=color)


def get_trigger_position(
    image_width,
    image_height,
    trigger_size,
    position,
):
    if position not in VALID_POSITIONS:
        raise ValueError(f"Invalid position: {position}")

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if (
        not isinstance(trigger_size, int)
        or isinstance(trigger_size, bool)
        or trigger_size <= 0
    ):
        raise ValueError("Trigger size must be a positive integer.")

    if trigger_size > image_width or trigger_size > image_height:
        raise ValueError("Trigger must be smaller than image.")

    if position == "bottom_right":
        return image_width - trigger_size, image_height - trigger_size
    if position == "center":
        return (
            (image_width - trigger_size) // 2,
            (image_height - trigger_size) // 2,
        )
    return 0, 0


def apply_trigger(image, trigger_size, trigger_color, position):
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image.")

    trigger_position = get_trigger_position(
        image.width,
        image.height,
        trigger_size,
        position,
    )
    trigger = create_trigger(trigger_size, trigger_color, image.mode)

    poisoned_image = image.copy()
    poisoned_image.paste(trigger, trigger_position)
    return poisoned_image


def poison_sample(image, is_poisoned, trigger_config):
    if not isinstance(is_poisoned, bool):
        raise TypeError("is_poisoned must be a boolean.")

    if not is_poisoned:
        return image

    if not isinstance(trigger_config, Mapping):
        raise TypeError("trigger_config must be a mapping.")

    trigger_size = trigger_config["size_px"]
    trigger_color = trigger_config["color"]
    position = trigger_config["position"]

    return apply_trigger(
        image,
        trigger_size,
        trigger_color,
        position
    )


def load_trigger_config(config_path, experiment_id):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, Mapping):
        raise ValueError("Invalid YAML configuration.")

    try:
        attack_config = config["attack"]
        trigger_config = attack_config["trigger"]
        target_config = attack_config["target"]
        experiment_config = config["experiments"][experiment_id]
    except KeyError as error:
        raise ValueError(f"Missing configuration key: {error.args[0]}") from error

    if not all(
        isinstance(section, Mapping)
        for section in (
            attack_config,
            trigger_config,
            target_config,
            experiment_config,
        )
    ):
        raise ValueError("Invalid trigger configuration structure.")

    trigger_type = trigger_config.get("type")
    if trigger_type != "square":
        raise ValueError(f"Unsupported trigger type: {trigger_type}")

    size_px = experiment_config.get(
        "trigger_size_px",
        trigger_config.get("default_size_px"),
    )
    if size_px == "default":
        size_px = trigger_config.get("default_size_px")

    position = experiment_config.get(
        "trigger_position",
        trigger_config.get("default_position"),
    )
    if position == "default":
        position = trigger_config.get("default_position")

    if size_px not in trigger_config.get("supported_sizes_px", []):
        raise ValueError(f"Unsupported trigger size: {size_px}")
    if position not in trigger_config.get("supported_positions", []):
        raise ValueError(f"Unsupported trigger position: {position}")
    if position not in VALID_POSITIONS:
        raise ValueError(f"Invalid position: {position}")

    color = trigger_config.get("color")
    target_label = target_config.get("label")
    target_class = target_config.get("class_name")

    if not isinstance(color, str) or not color:
        raise ValueError("Invalid trigger color.")

    if (
        not isinstance(size_px, int)
        or isinstance(size_px, bool)
        or size_px <= 0
    ):
        raise ValueError("Invalid trigger size.")

    if (
        not isinstance(target_label, int)
        or isinstance(target_label, bool)
        or target_label < 0
    ):
        raise ValueError("Invalid target label.")

    if not isinstance(target_class, str) or not target_class:
        raise ValueError("Invalid target class.")

    return {
        "type": trigger_type,
        "color": trigger_config.get("color"),
        "size_px": size_px,
        "position": position,
        "target_label": target_config.get("label"),
        "target_class": target_config.get("class_name"),
    }
