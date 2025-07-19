from ansys.fluent.core import launch_fluent, UIMode
import numpy as np
import os

# -------------------------
# 💡 Constants and Inputs
# -------------------------
print("🔧 Initializing constants...")
T_inf = 288.15  # Freestream temperature [K]
gamma = 1.4
R = 287.05      # Gas constant [J/kg-K]
mach_list = [0.2, 0.4, 0.6, 0.8, 1.0]



# -------------------------
# 🛠 Fluent Launch Config
# -------------------------
print("🚀 Setting Fluent launch path...")
os.environ["FLUENT_LAUNCHER_PATH"] = r"D:\Ansys\ANSYS Inc\v241\fluent\ntbin\win64\fluent.exe"
os.environ["ANSYSLI_LICEPATH"] = r"D:\Ansys\ANSYS Inc\Shared Files\Licensing"

print("FLUENT_LAUNCHER_PATH =", os.environ.get("FLUENT_LAUNCHER_PATH"))
print("ANSYSLI_LICEPATH     =", os.environ.get("ANSYSLI_LICEPATH"))
print("ANSYSLMD_LICENSE_FILE=", os.environ.get("ANSYSLMD_LICENSE_FILE"))


print("🧠 Launching Fluent session...")
session = launch_fluent(
    mode="solver",
    precision="double",
    processor_count=4,
    ui_mode=UIMode.GUI,
    start_transcript=True
)
print("✅ Fluent launched successfully!")

# -------------------------
# 📂 Load Case File
# -------------------------
case_path = "C:/Users/emcof/PEAR-Rocket(V1)_files/dp0/FFF/Fluent/FFF.1-Setup-Output.cas.h5"
print(f"📁 Reading case file: {case_path}")
session.file.read_case(case_path)
print("✅ Case file loaded.")

# -------------------------
# ⚙️ Physics Setup
# -------------------------
print("⚙️ Enabling energy equation...")
session.setup.models.energy = True

print("⚙️ Setting viscous model to k-omega SST...")
session.setup.models.viscous.k_omega_sst.enabled = True

print("🌬️ Configuring air material properties...")
air = session.setup.materials["air"]
air.density.definition = "ideal-gas"
air.viscosity.definition = "sutherland"
air.viscosity.suth_ref_viscosity = 1.716e-5
air.viscosity.suth_ref_temperature = 273.15
air.viscosity.suth_sutherland_constant = 110.4
print("✅ Material properties updated.")

# -------------------------
# 🔁 Mach Sweep Loop
# -------------------------
for mach in mach_list:
    print(f"\n🔄 Starting simulation for Mach {mach}...")

    # Freestream velocity
    a_inf = np.sqrt(gamma * R * T_inf)
    v_inf = mach * a_inf
    print(f"   ➤ Calculated velocity: {v_inf:.2f} m/s")

    # Boundary conditions (assumes 'inlet' is a pressure-far-field)
    print("   🔧 Setting boundary conditions...")
    inlet = session.setup.boundary_conditions.pressure_far_field["inlet"]
    inlet.mach_number = mach
    inlet.total_temperature = T_inf
    inlet.flow_direction.method = "components"
    inlet.flow_direction.x = 1.0
    inlet.flow_direction.y = 0.0
    inlet.flow_direction.z = 0.0

    print("   🧾 Setting reference values...")
    session.solution.reference_values.compute_from = "inlet"

    # Initialization
    print("   ⚡ Initializing flow field...")
    session.solution.initialization.hybrid_initialize()

    # Solve
    print("   🔄 Running solver iterations...")
    session.solution.run_calculation.iterate(iter_count=500)
    print("   ✅ Solution complete.")

    # Reporting
    print("   📊 Extracting drag force...")
    drag_report = session.solution.report_definitions.add_surface_integral(
        name=f"drag_force_m{mach}",
        field="pressure-force-x",
        surfaces=["cube_wall"]
    )
    drag_value = drag_report.get_data()
    print(f"🚀 Mach {mach} → Drag = {drag_value:.3f} N")

    # Save result
    result_file = f"result_mach_{mach:.2f}.dat.h5"
    session.file.write_data(result_file)
    print(f"💾 Result saved: {result_file}")
