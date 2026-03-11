import os
from pathlib import Path
from typing import List

import numpy as np
from Bio.PDB import Atom, Chain, Model, PDBIO, Residue, Structure
from scipy.interpolate import splprep, splev


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKETCH_CHECKPOINT = PROJECT_ROOT / "models" / "sketchcnn" / "ckpt" / "checkpoint"


def write_pdb(coords: np.ndarray, pdb_file: Path) -> None:
    structure = Structure.Structure("sketch")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    structure.add(model)
    model.add(chain)
    for i, coord in enumerate(coords, start=1):
        residue = Residue.Residue((" ", i, " "), "GLY", "")
        atom = Atom.Atom("CA", coord, 1.0, 1.0, " ", "CA", i, "C")
        residue.add(atom)
        chain.add(residue)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_file))


def add_intermediate_points(coords: np.ndarray, num_points_to_add: int = 1) -> np.ndarray:
    if len(coords) == 1:
        center_point = coords[0]
        random_points = center_point + np.random.randn(num_points_to_add, 3) * 0.1
        return np.vstack([center_point, random_points])

    new_coords = [coords[0]]
    for i in range(1, len(coords)):
        segment_length = np.linalg.norm(coords[i] - coords[i - 1])
        if segment_length > 0:
            for j in range(1, num_points_to_add + 1):
                fraction = j / (num_points_to_add + 1)
                new_point = coords[i - 1] + fraction * (coords[i] - coords[i - 1])
                new_coords.append(new_point)
        new_coords.append(coords[i])
    return np.array(new_coords)


def coords_spline(
    coords: np.ndarray,
    desired_length: int,
    mask_prob: float = 0.0,
    ss: List[float] = [5],
):
    coords = np.array(coords)
    if mask_prob > 0:
        keep_mask = np.random.random(len(coords)) >= mask_prob
        coords = coords[keep_mask]
    if len(coords) <= 3 and len(coords) >= 1:
        coords = add_intermediate_points(coords, num_points_to_add=4 - len(coords))
    elif len(coords) == 0:
        raise ValueError("coords must not be empty")

    tck, _ = splprep(coords.T, s=ss[0])
    spline_coords = np.array(splev(np.linspace(0, 1, desired_length), tck)).T
    lengths = np.sqrt(np.sum(np.diff(spline_coords, axis=0) ** 2, axis=1))

    def curvature(ts):
        der1 = np.array(splev(ts, tck, der=1))
        der2 = np.array(splev(ts, tck, der=2))
        num = np.linalg.norm(np.cross(der1, der2, axis=0), axis=0)
        den = np.linalg.norm(der1, axis=0) ** 3
        return num / den, der1.T

    curvatures, tangent_vectors = curvature(np.linspace(0, 1, desired_length))
    return spline_coords, curvatures, tangent_vectors, lengths


def get_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_normalized = axis / np.linalg.norm(axis)
    cross_product_matrix = np.array(
        [
            [0, -axis_normalized[2], axis_normalized[1]],
            [axis_normalized[2], 0, -axis_normalized[0]],
            [-axis_normalized[1], axis_normalized[0], 0],
        ]
    )
    return (
        np.eye(3)
        + np.sin(angle) * cross_product_matrix
        + (1 - np.cos(angle)) * np.dot(cross_product_matrix, cross_product_matrix)
    )


def rotate_point_on_circle(
    center: np.ndarray,
    normal_vector: np.ndarray,
    radius: float,
    angle: float,
) -> np.ndarray:
    normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)
    tangent_vector = np.cross(normal_vector_normalized, [1, 0, 0])
    if np.linalg.norm(tangent_vector) < 1e-6:
        tangent_vector = np.cross(normal_vector_normalized, [0, 1, 0])
    tangent_vector_normalized = tangent_vector / np.linalg.norm(tangent_vector)
    rotation_matrix = get_rotation_matrix(normal_vector_normalized, angle)
    rotated_tangent_vector = np.dot(rotation_matrix, tangent_vector_normalized)
    return center + radius * rotated_tangent_vector


def gene_helix(curve_points: np.ndarray, ca_num: int) -> List[np.ndarray]:
    radius = 2.25
    angle = 0
    interval = 10
    curve_points, _, tangent_vectors, _ = coords_spline(
        curve_points, (ca_num - 1) * interval + 1
    )
    ca_coords = []
    for i in range(0, len(curve_points), interval):
        angle = (angle + 100) % 360
        origin = curve_points[i]
        direction = tangent_vectors[i]
        ca_coords.append(
            rotate_point_on_circle(origin, direction, radius, np.radians(angle))
        )
    return ca_coords


def gene_loop(curve_points: np.ndarray, ca_num: int) -> List[np.ndarray]:
    curve_points, _, _, _ = coords_spline(curve_points, ca_num * 10)
    interval = max(len(curve_points) // ca_num, 1)
    return [curve_points[i] for i in range(0, len(curve_points), interval)][:ca_num]


def gene_beta(curve_points: np.ndarray, ca_num: int) -> List[np.ndarray]:
    radius = 3.25
    angle = 0
    interval = 4
    curve_points, _, tangent_vectors, _ = coords_spline(
        curve_points, (ca_num - 1) * interval + 1
    )
    ca_coords = []
    for i in range(0, len(curve_points), interval):
        angle = (angle + 180) % 360
        origin = curve_points[i]
        direction = tangent_vectors[i]
        pleat_offset = 0.75 * np.sin(i * np.pi / interval)
        origin = origin + pleat_offset * np.cross(direction, [1, 0, 0])
        ca_coords.append(
            rotate_point_on_circle(origin, direction, radius, np.radians(angle))
        )
    return ca_coords


def build_cnn_model(input_length: int, num_classes: int = 2, dropout_rate: float = 0.5):
    try:
        import tensorflow as tf
        if os.environ.get("DRAW_PROTEIN_USE_TF_GPU", "0") != "1":
            try:
                tf.config.set_visible_devices([], "GPU")
            except RuntimeError:
                pass
        from tensorflow.keras.layers import (
            BatchNormalization,
            Conv1D,
            Dense,
            Dropout,
            Flatten,
            Input,
            MaxPooling1D,
            Reshape,
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for mode=1 sketch generation."
        ) from exc

    inputs = Input(shape=(input_length, 1))
    x = BatchNormalization()(inputs)
    # Keep the original graph topology so the TensorFlow checkpoint matches.
    x = Conv1D(64, 3, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, activation="relu", padding="same")(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(input_length * num_classes, activation="softmax")(x)
    outputs = Reshape((input_length, num_classes))(outputs)
    return Model(inputs=inputs, outputs=outputs)


# def load_model(checkpoint_path: Path):
#     model = build_cnn_model(input_length=150, num_classes=2)
#     model.compile(
#         optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
#     )
#     model.load_weights(str(checkpoint_path))
#     return model

def load_model(checkpoint_path: Path):
    model = build_cnn_model(input_length=150, num_classes=2)

    # 只做推理，不要先 compile；否则老 checkpoint 里的 optimizer 状态会触发兼容问题
    status = model.load_weights(str(checkpoint_path))
    try:
        status.expect_partial()
    except Exception:
        pass

    return model


def get_count_list(label: np.ndarray, lengths: np.ndarray):
    count_list = []
    label_list = []
    length_list = []
    count = 0
    span_length = 0.0
    now = None
    for i, value in enumerate(label):
        if now is None:
            now = value
            count = 1
            span_length = 0.0
            continue
        if value == now:
            count += 1
            if i - 1 < len(lengths):
                span_length += lengths[i - 1]
        else:
            label_list.append(now)
            count_list.append(count)
            length_list.append(span_length)
            now = value
            count = 1
            span_length = 0.0
    label_list.append(now)
    count_list.append(count)
    length_list.append(span_length)
    return count_list, label_list, length_list


def flip_sequence(label: np.ndarray, threshold: int = 10) -> np.ndarray:
    label = np.array(label, copy=True)
    count_list, _, _ = get_count_list(label, np.zeros(len(label) - 1))
    offset = 0
    for count in count_list:
        if count < threshold:
            label[offset : offset + count] = 0
        offset += count

    count_list, _, _ = get_count_list(label, np.zeros(len(label) - 1))
    if len(count_list) == 2:
        if count_list[0] < threshold <= count_list[1]:
            label[: count_list[0]] = 1 - label[: count_list[0]]
        elif count_list[1] < threshold <= count_list[0]:
            label[-count_list[1] :] = 1 - label[-count_list[1] :]
        return label

    if len(count_list) > 2:
        if count_list[0] < threshold:
            label[: count_list[0]] = 1 - label[: count_list[0]]
        if count_list[-1] < threshold:
            label[-count_list[-1] :] = 1 - label[-count_list[-1] :]
        offset = count_list[0]
        for i in range(1, len(count_list) - 1):
            if count_list[i] < threshold <= count_list[i - 1] and threshold <= count_list[i + 1]:
                label[offset : offset + count_list[i]] = 1 - label[offset : offset + count_list[i]]
            offset += count_list[i]
    return label


def predict_label(model, spline_coords: np.ndarray, curvatures: np.ndarray) -> np.ndarray:
    spline_coords_chunks = np.array_split(spline_coords, len(spline_coords) // 150)
    curvature_chunks = np.array_split(curvatures, len(curvatures) // 150)
    labels = []
    for i in range(len(spline_coords_chunks)):
        curvature = curvature_chunks[i].reshape((1, 150, 1))
        pred = model.predict(curvature, verbose=0)
        labels.extend(np.argmax(pred, axis=2).reshape((150,)))
    return flip_sequence(np.array(labels), threshold=30)


def find_nearest_index(points: np.ndarray, target_point: np.ndarray) -> int:
    distances = np.linalg.norm(points - target_point, axis=1)
    return int(np.argmin(distances))


def modify_label(
    coords: np.ndarray,
    labels: np.ndarray,
    spline_coords: np.ndarray,
    spline_labels: np.ndarray,
) -> np.ndarray:
    segments = []
    start_idx = None
    current_label = None
    for i, label in enumerate(labels):
        if label in (0, 1):
            if current_label is None:
                start_idx = i
                current_label = label
            elif label != current_label:
                segments.append((start_idx, i - 1, current_label))
                start_idx = i
                current_label = label
        elif current_label is not None:
            segments.append((start_idx, i - 1, current_label))
            start_idx = None
            current_label = None
    if current_label is not None:
        segments.append((start_idx, len(labels) - 1, current_label))

    for start_idx, end_idx, label in segments:
        spline_start = find_nearest_index(spline_coords, coords[start_idx])
        spline_end = find_nearest_index(spline_coords, coords[end_idx])
        lo, hi = sorted((spline_start, spline_end))
        spline_labels[lo : hi + 1] = label
    return spline_labels


def get_helix_loop_ca_num(helix_length: float, loop_length: float):
    helix_ca = helix_length / 1.5 if helix_length else 0
    loop_ca = loop_length / 3.6 if loop_length else 0
    return helix_ca, loop_ca


def split_list(
    label: np.ndarray,
    spline_coords: np.ndarray,
    lengths: np.ndarray,
    tangent_vectors: np.ndarray,
):
    count_list, label_list, length_list = get_count_list(label, lengths)
    ca_num_list = []
    spline_coords_list = []
    tangent_vectors_list = []

    helix_length = sum(length for length, label_value in zip(length_list, label_list) if label_value == 1)
    loop_length = sum(length for length, label_value in zip(length_list, label_list) if label_value == 0)
    helix_ca, loop_ca = get_helix_loop_ca_num(helix_length, loop_length)

    spline_offset = 0
    for i, count in enumerate(count_list):
        if label_list[i] == 0:
            loop_num = int(loop_ca * length_list[i] / loop_length) if loop_length else 2
            ca_num_list.append(loop_num if loop_num > 1 else 2)
        else:
            helix_num = int(helix_ca * length_list[i] / helix_length) if helix_length else 2
            ca_num_list.append(helix_num if helix_num > 1 else 2)

        spline_coords_list.append(spline_coords[spline_offset : spline_offset + count])
        tangent_vectors_list.append(
            tangent_vectors[spline_offset : spline_offset + count]
        )
        spline_offset += count

    return ca_num_list, label_list, spline_coords_list, length_list, tangent_vectors_list


def predict_ca(
    ca_num_list: List[int],
    dssp_list: List[int],
    spline_coords_list: List[np.ndarray],
    tangent_vectors_list: List[np.ndarray],
) -> np.ndarray:
    ca_coords_list = []
    for i, ca_num in enumerate(ca_num_list):
        dssp = dssp_list[i]
        curve_coords = spline_coords_list[i]
        if dssp == 0:
            ca_coords = gene_loop(curve_coords, ca_num)
        elif dssp == 1:
            ca_coords = gene_helix(curve_coords, ca_num)
        else:
            ca_coords = gene_beta(curve_coords, ca_num)
        ca_coords_list.extend(ca_coords)
    return np.array(ca_coords_list)


def process_curve(curve_path: str, output_dir: str) -> int:
    curve_path = Path(curve_path)
    output_dir = Path(output_dir)
    if not SKETCH_CHECKPOINT.exists():
        raise FileNotFoundError(f"Sketch checkpoint not found: {SKETCH_CHECKPOINT}")

    output_dir.mkdir(parents=True, exist_ok=True)
    curve_name = curve_path.stem
    coords_labels = np.loadtxt(curve_path, delimiter=",")
    coords = coords_labels[:, :3]
    labels = np.full(len(coords), -1) if coords_labels.shape[1] == 3 else coords_labels[:, 3]
    spline_count = max(round(len(coords) * 30 / 150), 1) * 150

    spline_coords, curvatures, tangent_vectors, lengths = coords_spline(
        coords, desired_length=spline_count, mask_prob=0
    )
    model = load_model(SKETCH_CHECKPOINT)
    spline_labels = predict_label(model, spline_coords, curvatures)
    spline_labels = modify_label(coords, labels, spline_coords, spline_labels)
    spline_labels = flip_sequence(spline_labels, threshold=30)

    ca_num_list, dssp_list, spline_coords_list, _, tangent_vectors_list = split_list(
        spline_labels, spline_coords, lengths, tangent_vectors
    )
    ca_coords = predict_ca(
        ca_num_list, dssp_list, spline_coords_list, tangent_vectors_list
    )
    ca_num_total = len(ca_coords)

    np.savetxt(
        output_dir / f"{curve_name}-{ca_num_total}.npy",
        ca_coords,
        fmt="%.3f",
        delimiter=",",
    )
    write_pdb(ca_coords, output_dir / f"{curve_name}-{ca_num_total}.pdb")

    try:
        from tensorflow.keras import backend as K
        K.clear_session()
    except Exception:
        pass

    return ca_num_total
