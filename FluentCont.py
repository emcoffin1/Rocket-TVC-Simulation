# from ansys.fluent.core import launch_fluent, UIMode
import csv

from ansys.fluent.core import *
import numpy as np
import os
import io
from contextlib import redirect_stdout
aoa = 4

flow_field = [
    np.sin(np.deg2rad(aoa)),        # no roll/yaw — X is 0
    -np.cos(np.deg2rad(aoa)),       # into –Y (freestream)
    0.0                             # into –Z (pitch component)
]

drag_field = flow_field

lift_field = [
    np.cos(np.deg2rad(aoa)),        # no roll/yaw — X is 0
    np.sin(np.deg2rad(aoa)),       # into –Y (freestream)
    0.0                             # into –Z (pitch component)
]
# -------------------------
# 💡 Constants and Inputs
# -------------------------
print("🔧 Initializing constants...")
T_inf = 288.15  # Freestream temperature [K]
gamma = 1.4
R = 287.05      # Gas constant [J/kg-K]
start = 0.2
stop = 1.2
num = stop / 0.5
mach_list = np.linspace(start=start, stop=stop, num=6)
# mach_list = [0.4]
ROCKET_SURF = "rocket_body"
FARFIELD_SURF = "farfield"

# -------------------------
# 🛠 Fluent Launch Config
# -------------------------
print("🚀 Setting Fluent launch path and license env vars...")
os.environ["FLUENT_LAUNCHER_PATH"] = r"D:\Ansys\ANSYS Inc\v241\fluent\ntbin\win64\fluent.exe"
os.environ["ANSYSLI_LICEPATH"] = r"D:\Ansys\ANSYS Inc\Shared Files\Licensing"
os.environ["PYFLUENT_FLUENT_VERSION"] = "24.1.0"

print("FLUENT_LAUNCHER_PATH =", os.environ.get("FLUENT_LAUNCHER_PATH"))
print("ANSYSLI_LICEPATH     =", os.environ.get("ANSYSLI_LICEPATH"))
print("ANSYSLMD_LICENSE_FILE=", os.environ.get("ANSYSLMD_LICENSE_FILE"))


print("🧠 Launching Fluent session...")
session = launch_fluent(
    version="24.1.0",               # ✅ Explicit full license version
    mode="solver",
    precision="double",
    processor_count=6,
    ui_mode=UIMode.GUI,
    start_transcript=True
)
print(f"✅ Fluent launched successfully")
# print(session.settings.solution.child_names)
# quit()
# -------------------------
# 📂 Load Case File
# -------------------------
case_path = "C:/Users/emcof/PEAR-Rocket(V1)_files/dp0/FFF/Fluent/FFF.1-Setup-Output.cas.h5"
print(f"📁 Reading case file: {case_path}")
session.settings.file.read_case(file_name=case_path, pdf_file_name="")
print("✅ Case file loaded.")



# -------------------------
# ⚙️ Physics Setup
# -------------------------
print("⚙️ Enabling energy equation...")
session.settings.setup.models.energy.enabled = True

print("⚙️ Setting viscous model to k-omega SST...")
session.settings.setup.models.viscous.model = "k-omega"

# print("-"*15)
# print(dir(session.settings.setup.models.viscous.k_omega_model))
# print("-"*15)
session.settings.setup.models.viscous.k_omega_model = "sst"
session.settings.setup.models.viscous.options.curvature_correction = True
session.settings.setup.models.viscous.options.viscous_heating = True
session.settings.setup.models.viscous.options.compressibility_effects = True


print("🌬️ Configuring air material properties...")
air = session.settings.setup.materials.fluid["air"]
air.density.option = "ideal-gas"
air.viscosity.option = "sutherland"
air.viscosity.sutherland.reference_viscosity = 1.716e-5
air.viscosity.sutherland.reference_temperature = 273.15
air.viscosity.sutherland.effective_temperature = 110.4


print("✅ Material properties updated.")

session.settings.solution.methods.p_v_coupling.flow_scheme = "Coupled"


session.settings.solution.methods.warped_face_gradient_correction.enable = True
session.settings.solution.methods.high_order_term_relaxation.enable = True


# define a drag monitor on your zone

session.settings.solution.report_definitions.drag["drag"] = {}
drag_def = session.settings.solution.report_definitions.drag["drag"]
drag_def.force_vector       = drag_field
drag_def.zones              = [ROCKET_SURF]
drag_def.report_output_type = "Drag Force"
drag_def.create_output_parameter()

session.settings.solution.report_definitions.lift["lift"] = {}
lift_def = session.settings.solution.report_definitions.lift["lift"]
lift_def.force_vector       = lift_field
lift_def.zones              = [ROCKET_SURF]
lift_def.report_output_type = "Lift Force"
lift_def.create_output_parameter()

session.settings.solution.report_definitions.moment["moment"] = {}
mom_def = session.settings.solution.report_definitions.moment["moment"]
mom_def.report_type       = "moment"
mom_def.zones              = [ROCKET_SURF]
mom_def.mom_center  = [0.0,5.4862,0.0]
mom_def.mom_axis = [0.0,0.0,1.0]
mom_def.create_output_parameter()

session.settings.solution.report_definitions.surface['vel_mag'] = {}
vel_ref = session.settings.solution.report_definitions.surface['vel_mag']
vel_ref.report_type = "surface-areaavg"
vel_ref.field = "y-velocity"
vel_ref.surface_names = [FARFIELD_SURF]
vel_ref.create_output_parameter()


session.settings.solution.report_definitions.volume["dens_mag"] = {}
rho_def = session.settings.solution.report_definitions.volume["dens_mag"]
rho_def.report_type = "volume-average"
rho_def.field = "density"
rho_def.cell_zones = ["enclosure-enclosure"]
rho_def.create_output_parameter()

session.settings.solution.report_definitions.volume["reynolds"] = {}
reynolds = session.settings.solution.report_definitions.volume["reynolds"]
reynolds.report_type = "volume-average"
reynolds.field = "cell-reynolds-number"
reynolds.cell_zones = ["enclosure-enclosure"]
reynolds.create_output_parameter()

results = []

for mach in mach_list:
    print(f"\n🔄 Starting simulation for Mach {mach:.2f}…")

    # 1) Set the new farfield mach
    inlet = session.settings.setup.boundary_conditions.pressure_far_field[FARFIELD_SURF]
    inlet.set_state({
        "m": {"option": "value", "value": mach},
        "t": {"option": "value", "value": T_inf},
        "flow_direction": [
            {"option": "value", "value": flow_field[0]},   # 2 degree rotation into x (aoa 2)
            {"option": "value", "value": flow_field[1]},   # 2 degree rotation into x (aoa 2)
            {"option": "value", "value": flow_field[2]},
        ]
    })

    session.settings.solution.initialization.hybrid_initialize()
    session.settings.solution.run_calculation.iterate(iter_count=150)
    print("--Simulation Completed--")

    print("="*90)
    print("=" * 90)
    print(f"MACH: {mach}")
    vals = session.settings.solution.report_definitions.compute(
        report_defs=["vel_mag", "drag", "lift", "moment", "dens_mag", "reynolds"])
    results.append(vals)
    print("=" * 90)
    print("=" * 90)

with open(f"sweep_aoa{aoa}_", mode='w', newline='') as file:
    writer = csv.writer(file)
    for item in results:
        writer.writerow([item])
print("File saved!")
for x in results:
    print(x)

