import glob, subprocess
from IPython.display import Image

REPO = "/home/panchm/Code/ALEBGK4Public"   # <- where the repo is
K1   = "/home/panchm/Code/ALEBGK4Public/output/cavitybodymoving_2D_Nx60_Nv30_kn001"  # <- your VTK folders
#K2   = "/path/to/output/cavitybodymoving_2D_Nx60_Nv30_kn01"

def panels(folder, kn):
    vtks = sorted(glob.glob(f"{folder}/output_*.vtk"))
    b = f"{folder}/bodies.csv"
    mid, last = vtks[len(vtks)//2], vtks[-1]
    return [f"VTK={mid},BODIES={b},LABEL={kn} (half orbit)",
            f"VTK={last},BODIES={b},LABEL={kn} (full orbit)"]

specs = panels(K1, "Kn=0.01")# + panels(K2, "Kn=0.1")
subprocess.run(["python3", f"{REPO}/postprocessing/plot_cavity_body_field.py",
                "cavity_body_field.png", *specs], check=True)
Image("cavity_body_field.png")