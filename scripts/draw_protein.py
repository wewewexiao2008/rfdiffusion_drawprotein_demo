import argparse
import datetime
import glob
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from Bio import PDB
from Bio.PDB import Atom, Chain, Model, PDBIO, PDBParser, Residue, Structure
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_INFERENCE = PROJECT_ROOT / "scripts" / "run_inference.py"
DEFAULT_WORKDIR = PROJECT_ROOT / "runs"

sys.path.insert(0, str(PROJECT_ROOT))
from sketch_process import process_curve


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def write_pdb(coords: np.ndarray, pdb_file: Path) -> None:
    structure = Structure.Structure("example")
    model = Model.Model(0)
    structure.add(model)
    chain = Chain.Chain("A")
    model.add(chain)
    for i, coord in enumerate(coords, start=1):
        residue = Residue.Residue((" ", i, " "), "GLY", "")
        atom = Atom.Atom("CA", coord, 1.0, 1.0, " ", "CA", i, "C")
        residue.add(atom)
        chain.add(residue)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_file))


def reindex(pdb_file: str, output_dir: Path, startwith: int = 1) -> Path:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", str(pdb_file))

    for model in structure:
        for chain in model:
            residue_counter = startwith
            residues_to_remove = []
            for residue in chain:
                if residue.id[0] == " ":
                    residue.id = (residue.id[0], residue_counter, residue.id[2])
                    residue_counter += 1
                else:
                    residues_to_remove.append(residue)
            for residue in residues_to_remove:
                chain.detach_child(residue.id)

    writer = PDB.PDBIO()
    writer.set_structure(structure)
    output_path = output_dir / f"reindexed_{Path(pdb_file).stem}.pdb"
    writer.save(str(output_path))
    return output_path


def translate_pdbs_to_origin(pdb_path: Path, npy_path: Path) -> Path:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", str(pdb_path))
    coords = np.array([atom.coord for atom in structure.get_atoms() if atom.element != "H"])
    centroid = coords.mean(axis=0)
    translation_vector = -centroid

    for atom in structure.get_atoms():
        atom.coord += translation_vector

    translated_coords = np.loadtxt(npy_path, delimiter=",")[:, :3] + translation_vector
    translated_path = pdb_path.parent / "binder_translated.npy"
    np.savetxt(translated_path, translated_coords, delimiter=",")

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))
    return translated_path


def find_closest(coord: np.ndarray, input_pdb: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(input_pdb))
    closest_residue = None
    min_distance = float("inf")
    num_residues = 0

    for model in structure:
        for chain in model:
            num_residues = len(list(chain.get_residues()))
            for residue in chain.get_residues():
                for atom in residue.get_atoms():
                    distance = np.linalg.norm(coord - atom.get_coord())
                    if distance < min_distance:
                        min_distance = distance
                        closest_residue = residue

    if closest_residue is None:
        raise ValueError(f"No residues found in {input_pdb}")

    residue_id = closest_residue.get_id()[1]
    chain_id = closest_residue.get_parent().id
    print(f"Closest residue: {residue_id} in chain {chain_id}")
    return chain_id, residue_id, num_residues


def calculate_hotspot(input_pdb: Path, sketch_path: Path, cutoff=None):
    sketch_coords = np.loadtxt(sketch_path, delimiter=",")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(input_pdb))
    residue_distances = []
    configmap = None

    for model in structure:
        for chain in model:
            residues = list(chain.get_residues())
            if residues:
                start_residue = residues[0].get_id()[1] + 1
                end_residue = residues[-1].get_id()[1]
                configmap = f"{chain.get_id()}{start_residue}-{end_residue}/0"
            for residue in chain:
                min_distance = float("inf")
                for atom in residue:
                    distances = np.linalg.norm(sketch_coords - atom.get_coord(), axis=1)
                    min_distance = min(min_distance, np.min(distances))
                residue_distances.append((f"{chain.get_id()}{residue.get_id()[1]}", min_distance))

    closest_residues = sorted(residue_distances, key=lambda x: x[1])[:10]
    residue_ids = [
        residue_id
        for residue_id, distance in closest_residues
        if cutoff is None or distance < cutoff
    ]
    return configmap, "[" + ",".join(residue_ids) + "]"


def calculate_drag(input_pdb: Path, sketch_path: Path, position: int = 2) -> str:
    sketch_coords = np.loadtxt(sketch_path, delimiter=",")
    seqlen = len(sketch_coords)
    if position == 0:
        chain_id, residue_id, num_residues = find_closest(sketch_coords[-1], input_pdb)
        configmap = f"[{seqlen}-{seqlen}/{chain_id}{residue_id + 1}-{num_residues}]"
        print(configmap)
        return configmap
    if position == 1:
        chain_id_1, residue_id_1, num_residues = find_closest(sketch_coords[-1], input_pdb)
        chain_id_0, residue_id_0, _ = find_closest(sketch_coords[0], input_pdb)
        return f"[{chain_id_0}1-{residue_id_0 - 1}/{seqlen}-{seqlen}/{chain_id_1}{residue_id_1 + 1}-{num_residues}]"

    chain_id, residue_id, _ = find_closest(sketch_coords[0], input_pdb)
    configmap = f"[{chain_id}1-{residue_id - 1}/{seqlen}-{seqlen}]"
    print(configmap)
    return configmap


def run_inference(*overrides: str) -> None:
    cmd = [sys.executable, str(RUN_INFERENCE), *overrides]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main(args) -> None:
    work_root = Path(args.work_dir)
    work_dir = work_root / args.name
    if args.debug:
        shutil.rmtree(work_dir, ignore_errors=True)
    if work_dir.exists():
        raise FileExistsError(f"Output directory already exists: {work_dir}")
    work_dir.mkdir(parents=True)

    curve_path = None
    curve_coords = None
    lamda_0 = 1
    lamda_1 = 1
    lamda_2 = 0
    if args.curve_path:
        src_curve = Path(args.curve_path).resolve()
        if not src_curve.exists():
            raise FileNotFoundError(f"Curve file does not exist: {src_curve}")
        curve_path = Path(shutil.copy(src_curve, work_dir))
        curve_coords_labels = np.loadtxt(curve_path, delimiter=",")
        curve_coords = curve_coords_labels[:, :3]
        write_pdb(curve_coords, curve_path.with_suffix(".pdb"))
        lamda_0 = args.lamda_0
        lamda_1 = args.lamda_1
        lamda_2 = args.lamda_2

    sketch_dir = work_dir / "sketchs"
    rfdiffusion_dir = work_dir / "rfdiffusion"
    design_dir = work_dir / "designs"
    sketch_dir.mkdir()
    rfdiffusion_dir.mkdir()
    design_dir.mkdir()

    print("[1] Parametric sampling Protein Sketch ========= ")
    if args.task == "unconditional":
        if not args.length:
            raise ValueError("Length is required for unconditional mode")
        run_inference(
            f"inference.output_prefix={rfdiffusion_dir / str(args.length)}",
            "inference.lamda_0=1",
            "inference.lamda_1=0",
            "inference.lamda_2=0",
            f"contigmap.contigs=[{args.length}-{args.length}]",
            f"inference.num_designs={args.num_designs}",
        )
    else:
        if curve_path is None or curve_coords is None:
            raise ValueError("Curve path is required for this task")

        if args.mode == 1:
            process_curve(str(curve_path), str(sketch_dir))
        else:
            basename = curve_path.stem
            np.savetxt(
                sketch_dir / f"{basename}-{len(curve_coords)}.npy",
                curve_coords,
                fmt="%.3f",
                delimiter=",",
            )
            write_pdb(curve_coords, sketch_dir / f"{basename}-{len(curve_coords)}.pdb")

        print("[2] Generating Sketch-guided Protein using RFDiffusion ========= ")
        sketch_paths = sorted(glob.glob(str(sketch_dir / "*.npy")))

        if args.task == "binder":
            if not args.input_pdb:
                raise ValueError("Input pdb is required for binder mode")
            reindexed_input = reindex(args.input_pdb, work_dir, 1)
            for sketch_path_str in tqdm(sketch_paths, desc="RFDiffusion", disable=False):
                sketch_path = Path(sketch_path_str)
                configmap, hotspot = calculate_hotspot(reindexed_input, sketch_path)
                basename = sketch_path.stem
                length = basename.split("-")[-1]
                translated_npy = translate_pdbs_to_origin(reindexed_input, sketch_path)
                run_inference(
                    f"inference.output_prefix={rfdiffusion_dir / basename}",
                    "inference.lamda_0=2",
                    "inference.lamda_1=4",
                    f"inference.lamda_2={lamda_2}",
                    f"inference.cX_path={translated_npy}",
                    f"inference.input_pdb={reindexed_input}",
                    f"ppi.hotspot_res={hotspot}",
                    f"contigmap.contigs=[{configmap} {length}-{length}]",
                    f"inference.num_designs={args.num_designs}",
                    "inference.method=6",
                )
        elif args.task == "denovo":
            for sketch_path_str in tqdm(sketch_paths, desc="RFDiffusion", disable=False):
                sketch_path = Path(sketch_path_str)
                basename = sketch_path.stem
                length = basename.split("-")[-1]
                run_inference(
                    f"inference.output_prefix={rfdiffusion_dir / basename}",
                    f"diffuser.T={args.diffuser_T}",
                    f"inference.cX_path={sketch_path}",
                    f"inference.method={args.method}",
                    f"inference.time={args.time}",
                    f"inference.lamda_0={lamda_0}",
                    f"inference.lamda_1={lamda_1}",
                    f"inference.lamda_2={lamda_2}",
                    f"contigmap.contigs=[{length}-{length}]",
                    f"inference.num_designs={args.num_designs}",
                )
        elif args.task == "drag":
            if not args.input_pdb:
                raise ValueError("Input pdb is required for drag mode")
            reindexed_input = reindex(args.input_pdb, work_dir, 1)
            for sketch_path_str in tqdm(sketch_paths, desc="RFDiffusion", disable=False):
                sketch_path = Path(sketch_path_str)
                basename = sketch_path.stem
                configmap = calculate_drag(reindexed_input, sketch_path, args.position)
                run_inference(
                    f"inference.output_prefix={rfdiffusion_dir / basename}",
                    "inference.lamda_0=2",
                    "inference.lamda_1=4",
                    "inference.lamda_2=1",
                    f"inference.cX_path={sketch_path}",
                    f"inference.input_pdb={reindexed_input}",
                    f"contigmap.contigs={configmap}",
                    f"inference.num_designs={args.num_designs}",
                    "inference.method=5",
                )

    for protein_path in glob.glob(str(rfdiffusion_dir / "*.pdb")):
        shutil.copy(protein_path, design_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument("-c", "--curve_path", type=str, help="Path to the curve file")
    parser.add_argument("-i", "--input_pdb", type=str, help="Input target pdb")
    parser.add_argument("-n", "--name", type=str, default=time_now, help="Run name")
    parser.add_argument("-m", "--method", type=int, default=1, help="Conditioning method")
    parser.add_argument(
        "-w",
        "--work_dir",
        type=str,
        default=str(DEFAULT_WORKDIR),
        help="Directory where new runs will be created",
    )
    parser.add_argument("-d", "--debug", type=parse_bool, default=True, help="Delete the run dir first")
    parser.add_argument("-dt", "--diffuser_T", type=int, default=50, help="Diffuser steps")
    parser.add_argument("-p", "--position", type=int, default=2, help="Drag position")
    parser.add_argument("-mo", "--mode", type=int, default=1, choices=[1, 2], help="1=curve->sketch, 2=direct sketch")
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="denovo",
        choices=["denovo", "binder", "unconditional", "drag", "partial-diffuse", "motif-scaffoding"],
        help="Task type",
    )
    parser.add_argument("-T", "--time", type=int, default=10, help="Method time parameter")
    parser.add_argument("-num", "--num_designs", type=int, default=1, help="Number of designs")
    parser.add_argument("-l0", "--lamda_0", type=float, default=6.0, help="lamda_0")
    parser.add_argument("-l1", "--lamda_1", type=float, default=2.0, help="lamda_1")
    parser.add_argument("-l2", "--lamda_2", type=float, default=0.0, help="lamda_2")
    parser.add_argument("-l", "--length", type=int, help="Protein length for unconditional mode")
    main(parser.parse_args())
