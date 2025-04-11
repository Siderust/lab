import os
import pandas as pd
import matplotlib.pyplot as plt

lab_root = os.environ.get("SIDERUST_LAB_ROOT", ".")

def compare_datasets(astropy_file, siderust_file, body_name, frame_name):
    """
    Loads two CSV files (Astropy vs. Siderust) for the same body and frame,
    merges/join on JD if needed, computes error stats, and returns a DataFrame
    with the comparison columns:
        - jd
        - dx, dy, dz
        - dist_error
    Prints summary statistics to stdout.
    """

    # Load data
    df_astropy = pd.read_csv(astropy_file)
    df_siderust = pd.read_csv(siderust_file)

    # Rename columns for consistency if needed
    df_astropy.rename(columns={"JD": "jd", "X_AU": "x", "Y_AU": "y", "Z_AU": "z"}, inplace=True)
    df_siderust.rename(columns={"JD": "jd", "X_AU": "x", "Y_AU": "y", "Z_AU": "z"}, inplace=True)

    # Merge or combine
    if not df_astropy["jd"].equals(df_siderust["jd"]):
        print(f"[WARNING] JD columns differ for {body_name} ({frame_name}). Merging on JD...")
        df = pd.merge(df_astropy, df_siderust, on="jd", suffixes=("_astropy","_siderust"))
        # Create dx, dy, dz from merged columns
        df["dx"] = df["x_siderust"] - df["x_astropy"]
        df["dy"] = df["y_siderust"] - df["y_astropy"]
        df["dz"] = df["z_siderust"] - df["z_astropy"]
    else:
        # Rows match up
        df = df_astropy.copy()
        df["x_siderust"] = df_siderust["x"]
        df["y_siderust"] = df_siderust["y"]
        df["z_siderust"] = df_siderust["z"]
        df["dx"] = df["x_siderust"] - df["x"]
        df["dy"] = df["y_siderust"] - df["y"]
        df["dz"] = df["z_siderust"] - df["z"]

    # Compute 3D distance error
    df["dist_error"] = (df["dx"]**2 + df["dy"]**2 + df["dz"]**2) ** 0.5

    # Basic stats
    mean_dx = df["dx"].mean()
    mean_dy = df["dy"].mean()
    mean_dz = df["dz"].mean()

    std_dx = df["dx"].std()
    std_dy = df["dy"].std()
    std_dz = df["dz"].std()

    max_dx = df["dx"].abs().max()
    max_dy = df["dy"].abs().max()
    max_dz = df["dz"].abs().max()

    rms_3d = (df["dist_error"].pow(2).mean())**0.5
    max_3d = df["dist_error"].max()

    # Print summary
    print(f"==== Comparison for {body_name} ({frame_name}) ====")
    print(f" Number of rows: {len(df)}")
    print(f" Mean error in X/Y/Z (AU): {mean_dx:.3e}, {mean_dy:.3e}, {mean_dz:.3e}")
    print(f" Std  error in X/Y/Z (AU): {std_dx:.3e}, {std_dy:.3e}, {std_dz:.3e}")
    print(f" Max  error in X/Y/Z (AU): {max_dx:.3e}, {max_dy:.3e}, {max_dz:.3e}")
    print(f" RMS 3D error     (AU)   : {rms_3d:.3e}")
    print(f" Max 3D error     (AU)   : {max_3d:.3e}")
    print("")

    return df

def plot_comparison(df, body_name, frame_name):
    """
    Plots dx, dy, dz, and dist_error vs. JD in four separate figures.
    Each figure is saved to a PNG file (or just shown).
    """

    save_path = lab_root+f"plots/{body_name}_{frame_name}"

    # dx
    plt.figure()
    plt.plot(df["jd"], df["dx"])
    plt.xlabel("Julian Date")
    plt.ylabel("dx (AU)")
    plt.title(f"{body_name} ({frame_name}) - ΔX vs. JD")
    plt.savefig(f"{save_path}_dx.png")
    plt.close()

    # dy
    plt.figure()
    plt.plot(df["jd"], df["dy"])
    plt.xlabel("Julian Date")
    plt.ylabel("dy (AU)")
    plt.title(f"{body_name} ({frame_name}) - ΔY vs. JD")
    plt.savefig(f"{save_path}_dy.png")
    plt.close()

    # dz
    plt.figure()
    plt.plot(df["jd"], df["dz"])
    plt.xlabel("Julian Date")
    plt.ylabel("dz (AU)")
    plt.title(f"{body_name} ({frame_name}) - ΔZ vs. JD")
    plt.savefig(f"{save_path}_dz.png")
    plt.close()

    # dist_error
    plt.figure()
    plt.plot(df["jd"], df["dist_error"])
    plt.xlabel("Julian Date")
    plt.ylabel("3D Error (AU)")
    plt.title(f"{body_name} ({frame_name}) - 3D Distance Error vs. JD")
    plt.savefig(f"{save_path}_3d.png")
    plt.close()

def main():
    """
    Compare multiple CSV pairs for each planet, for each coordinate frame:
      - Helio (heliocentric)
      - Geo (geocentric)
      - Possibly ICRS, if needed (remove if not desired)
    Print stats and generate plots.
    """

    bodies = ["sun", "moon", "mercury", "venus", "earth", "mars",
              "jupiter", "saturn", "uranus", "neptune"]

    # If you only want helio and geo, keep the below frames, or add "icrs" if you like.
    frames = ["helio", "geo"]  # or ["helio", "geo", "icrs"]

    for frame in frames:
        # CSV file name templates
        astropy_template = os.path.join(lab_root, "astropy", "dataset", frame, "{body}_{frame}.csv")
        siderust_template = os.path.join(lab_root, "siderust", "dataset", frame, "{body}_{frame}.csv")

        print(f"\n==== Now comparing {frame.upper()} coordinates ====")
        for body in bodies:
            astropy_file = astropy_template.format(body=body, frame=frame)
            siderust_file = siderust_template.format(body=body, frame=frame)

            if not os.path.exists(astropy_file):
                print(f"[ERROR] Astropy file not found for {body} ({frame}): {astropy_file}")
                continue
            if not os.path.exists(siderust_file):
                print(f"[ERROR] Siderust file not found for {body} ({frame}): {siderust_file}")
                continue

            df_result = compare_datasets(astropy_file, siderust_file, body, frame)
            plot_comparison(df_result, body, frame)

if __name__ == "__main__":
    main()
