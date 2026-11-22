# run_sweep.py
import os
import sys
import numpy as np
import pandas as pd
import yaml

# Enable native ANSI color support in Windows terminal without any external package
os.system('')

CLR_RESET   = "\033[0m"
CLR_BOLD    = "\033[1m"
CLR_CYAN    = "\033[1;36m"
CLR_GREEN   = "\033[1;32m"
CLR_YELLOW  = "\033[1;33m"
CLR_BLUE    = "\033[1;34m"
CLR_MAGENTA = "\033[1;35m"
CLR_RED     = "\033[1;31m"
CLR_GRAY    = "\033[90m"

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

    fsp_file_path = os.path.abspath("Silicon-on-MgF2-Metasurface.fsp")
    if not os.path.exists(fsp_file_path):
        print(f"{CLR_RED}[ERROR]{CLR_RESET} Simulation file not found: {fsp_file_path}")
        return

    # ---- Resume logic ----
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"{CLR_CYAN}=" * 60)
    print(f" {CLR_BOLD}[SWEEP]{CLR_RESET} {CLR_CYAN}Silicon-on-MgF2 Metasurface  Automated Sweep{CLR_RESET}")
    print(f"{CLR_CYAN}=" * 60 + f"{CLR_RESET}")

    if os.path.exists(output_csv):
        existing_df  = pd.read_csv(output_csv)
        start_idx    = len(existing_df)
        header_written = True
        print(f"{CLR_YELLOW}[RESUME]{CLR_RESET} Existing dataset found: {CLR_BOLD}{start_idx}{CLR_RESET} samples completed.")
        print(f"{CLR_YELLOW}[RESUMING]{CLR_RESET} Resuming from sample {CLR_BOLD}{start_idx + 1}/{num_samples}{CLR_RESET}...")
    else:
        start_idx      = 0
        header_written = False
        print(f"{CLR_BLUE}[INIT]{CLR_RESET}   No existing dataset found. Starting fresh sweep.")

    if start_idx >= num_samples:
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Target reached! Dataset is already complete.")
        return

    # ---- Lumerical session ----
    print(f"{CLR_MAGENTA}[FDTD]{CLR_RESET}   Launching Lumerical engine with: {CLR_BOLD}{os.path.basename(fsp_file_path)}{CLR_RESET}")
    fd = lumapi.FDTD(filename=fsp_file_path)

    w_start = config.get('simulation', {}).get('wavelength_start_nm', 5500) * 1e-9
    w_end   = config.get('simulation', {}).get('wavelength_end_nm',   7000) * 1e-9

    print(f"{CLR_BLUE}[CONFIG]{CLR_RESET} Wavelength range: {CLR_BOLD}{w_start*1e6:.2f} um to {w_end*1e6:.2f} um{CLR_RESET} | Points: {CLR_BOLD}{w_points}{CLR_RESET}")

    for source_name in ["source", "Plane Source"]:
        try:
            fd.setnamed(source_name, "wavelength start", w_start)
            fd.setnamed(source_name, "wavelength stop",  w_end)
            print(f"  {CLR_GRAY}|--{CLR_RESET} Source configured: '{source_name}'")
            break
        except Exception:
            continue

    for monitor_name in ["R", "Reflection Monitor"]:
        try:
            fd.setnamed(monitor_name, "override global monitor settings", True)
            fd.setnamed(monitor_name, "frequency points", w_points)
            print(f"  {CLR_GRAY}|--{CLR_RESET} Monitor configured: '{monitor_name}' ({w_points} pts)")
            break
        except Exception:
            continue

    print(f"{CLR_GRAY}-" * 60 + f"{CLR_RESET}")

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

        print(f"{CLR_GREEN}[SIMULATION RUNNNING]{CLR_RESET}  Sample {i+1}/{num_samples} | Simulating: H={H*1e6:.3f}um, R1={R1*1e6:.3f}um, R2={R2*1e6:.3f}um, Px={Px*1e6:.3f}um, Py={Py*1e6:.3f}um")
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

        print(f"  {CLR_GRAY}|--{CLR_RESET} {CLR_GREEN}Saved{CLR_RESET} -> {output_csv} (Sample {i+1}/{num_samples})")

    fd.close()
    print(f"{CLR_CYAN}=" * 60 + f"{CLR_RESET}")
    print(f"{CLR_GREEN}[DONE]{CLR_RESET} Sweep complete! Total {CLR_BOLD}{num_samples}{CLR_RESET} samples generated in '{output_csv}'.")
    print(f"{CLR_CYAN}=" * 60 + f"{CLR_RESET}")

if __name__ == "__main__":
    run_automated_sweep()
