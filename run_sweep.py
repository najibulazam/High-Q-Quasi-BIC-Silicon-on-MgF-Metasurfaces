# run_sweep.py
import os
import sys
import numpy as np
import pandas as pd
import yaml

try:
    import lumapi
except ImportError:
    lum_paths = [
        r"C:\Program Files\Lumerical\v241\api\python",
        r"C:\Program Files\Lumerical\v232\api\python",
        r"C:\Program Files\ANSYS Inc\v241\Lumerical\api\python",
        r"C:\Program Files\ANSYS Inc\v232\Lumerical\api\python"
    ]
    for path in lum_paths:
        if os.path.exists(path):
            sys.path.append(path)
            try:
                import lumapi
                break
            except ImportError:
                continue
    else:
        raise ImportError("lumapi path not found. Please check your Lumerical installation.")

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def run_automated_sweep():
    num_samples = config.get('dataset', {}).get('num_samples', 11000)
    w_points    = config.get('simulation', {}).get('wavelength_points', 1000)
    output_csv  = config.get('dataset', {}).get('output_csv', 'dataset/spectral_data.csv')

    fsp_file_path = os.path.abspath("MgF2-Substrate-Structure.fsp")
    if not os.path.exists(fsp_file_path):
        print(f"[ERROR] Simulation file not found: {fsp_file_path}")
        return

    # ---- Resume logic ----
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print("=" * 60)
    print(" [SWEEP] MgF2 Substrate Metasurface Automated Sweep")
    print("=" * 60)

    if os.path.exists(output_csv):
        existing_df  = pd.read_csv(output_csv)
        start_idx    = len(existing_df)
        header_written = True
        print(f"[RESUME] Existing dataset found: {start_idx} samples completed.")
        print(f"[RESUME] Resuming from sample {start_idx + 1}/{num_samples}...")
    else:
        start_idx      = 0
        header_written = False
        print("[INIT]   No existing dataset found. Starting fresh sweep.")

    if start_idx >= num_samples:
        print("[DONE]   Target reached! Dataset is already complete.")
        return

    # ---- Lumerical session ----
    print(f"[FDTD]   Launching Lumerical engine with: {os.path.basename(fsp_file_path)}...")
    fd = lumapi.FDTD(filename=fsp_file_path)

    w_start = config.get('simulation', {}).get('wavelength_start_nm', 5600) * 1e-9
    w_end   = config.get('simulation', {}).get('wavelength_end_nm',   6600) * 1e-9

    print(f"[CONFIG] Wavelength range: {w_start*1e6:.2f} um to {w_end*1e6:.2f} um | Points: {w_points}")

    for source_name in ["source", "Plane Source"]:
        try:
            fd.setnamed(source_name, "wavelength start", w_start)
            fd.setnamed(source_name, "wavelength stop",  w_end)
            print(f"  |-- Source configured: '{source_name}'")
            break
        except Exception:
            continue

    for monitor_name in ["R", "Reflection Monitor"]:
        try:
            fd.setnamed(monitor_name, "override global monitor settings", True)
            fd.setnamed(monitor_name, "frequency points", w_points)
            print(f"  |-- Monitor configured: '{monitor_name}' ({w_points} pts)")
            break
        except Exception:
            continue

    print("-" * 60)

    # ---- Sweep loop ----
    for i in range(start_idx, num_samples):
        fd.switchtolayout()

        H  = float(np.random.uniform(0.78e-6, 0.82e-6))
        Px = float(np.random.uniform(4.02e-6, 4.07e-6))
        Py = float(np.random.uniform(2.32e-6, 2.36e-6))
        R1 = float(np.random.uniform(0.31e-6, 0.46e-6))
        R2 = float(np.random.uniform(1.115e-6, 1.130e-6))

        fd.setnamed("FDTD", "x span", Px)
        fd.setnamed("FDTD", "y span", Py)

        fd.setnamed("Left ellipse",  "radius",   R1)
        fd.setnamed("Left ellipse",  "radius 2", R2)
        fd.setnamed("Left ellipse",  "z span",   H)

        fd.setnamed("Right ellipse", "radius",   R1)
        fd.setnamed("Right ellipse", "radius 2", R2)
        fd.setnamed("Right ellipse", "z span",   H)

        print(f"[RUN]  Sample {i+1}/{num_samples} | Simulating: H={H*1e6:.3f}um, R1={R1*1e6:.3f}um...")
        fd.run()

        R_result       = fd.getresult("R", "T")
        reflection     = np.real(R_result['T']).flatten()
        wavelengths_um = (R_result['lambda'] * 1e6).flatten()

        row_data = {
            "H_um":  H  * 1e6,
            "Px_um": Px * 1e6,
            "Py_um": Py * 1e6,
            "R1_um": R1 * 1e6,
            "R2_um": R2 * 1e6,
        }
        for idx in range(min(w_points, len(reflection))):
            row_data[f"w_{idx}"] = float(wavelengths_um[idx])
            row_data[f"r_{idx}"] = float(reflection[idx])

        row_df = pd.DataFrame([row_data])
        if not header_written:
            row_df.to_csv(output_csv, index=False, mode='w', header=True)
            header_written = True
        else:
            row_df.to_csv(output_csv, index=False, mode='a', header=False)

        print(f"  |-- Saved -> {output_csv} (Sample {i+1}/{num_samples})")

    fd.close()
    print("=" * 60)
    print(f"[DONE] Sweep complete! Total {num_samples} samples generated in '{output_csv}'.")
    print("=" * 60)

if __name__ == "__main__":
    run_automated_sweep()