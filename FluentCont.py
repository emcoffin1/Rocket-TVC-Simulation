# from ansys.fluent.core import launch_fluent, UIMode
from ansys.fluent.core import *
import numpy as np
import os

# -------------------------
# 💡 Constants and Inputs
# -------------------------
print("🔧 Initializing constants...")
T_inf = 288.15  # Freestream temperature [K]
gamma = 1.4
R = 287.05      # Gas constant [J/kg-K]
mach_list = [0.4, 0.6]
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
drag_def.force_vector       = [0.0, 1.0, 0.0]
drag_def.zones              = [ROCKET_SURF]
drag_def.report_output_type = "Drag Force"
drag_def.create_output_parameter()

session.settings.solution.report_definitions.lift["lift"] = {}
lift_def = session.settings.solution.report_definitions.lift["lift"]
lift_def.force_vector       = [1.0, 0.0, 0.0]
lift_def.zones              = [ROCKET_SURF]
lift_def.report_output_type = "Lift Force"
lift_def.create_output_parameter()

session.settings.solution.report_definitions.moment["moment"] = {}
mom_def = session.settings.solution.report_definitions.moment["moment"]
mom_def.report_type       = "moment"
mom_def.zones              = [ROCKET_SURF]
mom_def.mom_center  = [0.0,0.0,0.0]
mom_def.mom_axis = [0.0,0.0,1.0]
mom_def.create_output_parameter()

session.settings.solution.report_definitions.surface['vel_mag'] = {}
vel_ref = session.settings.solution.report_definitions.surface['vel_mag']
vel_ref.report_type = "surface-areaavg"
vel_ref.field = "y-velocity"
vel_ref.surface_names = [FARFIELD_SURF]
vel_ref.create_output_parameter()

results = []
# -------------------------
# 🔁 Mach Sweep Loop
# -------------------------
for mach in mach_list:
    print(f"\n🔄 Starting simulation for Mach {mach}...")

    # Freestream velocity
    a_inf = np.sqrt(gamma * R * T_inf)
    v_inf = mach * a_inf

    # Boundary conditions (assumes 'inlet' is a pressure-far-field)
    print("   🔧 Setting boundary conditions...")
    inlet = session.settings.setup.boundary_conditions.pressure_far_field[FARFIELD_SURF]

    inlet.set_state({
        "m": {"option": "value", "value": mach},
        "t": {"option": "value", "value": 288.13},
        "flow_direction": [
            {"option": "value", "value": 0},
            {"option": "value", "value": -1},
            {"option": "value", "value": 0},
        ]
    })


    session.settings.setup.general.operating_conditions.operating_pressure = 101325


    # print("   ⚡ Initializing flow field...")
    session.settings.solution.initialization.hybrid_initialize()

    session.settings.solution.run_calculation.pseudo_time_settings.time_step_method.set_state({
        'time_step_method': 'automatic', 'length_scale_methods': 'conservative', 'time_step_size_scale_factor': 0.1
    })
    session.settings.solution.run_calculation.iter_count = 1


    # print(session.settings.solution.report_definitions.get_state())

    # session.settings.solution.report_definitions.set_state({
    #     'surface': {
    #         'vel-magnitude': {
    #             'name': 'vel-magnitude',
    #             'report_type': 'surface-areaavg',
    #             'field': 'velocity-magnitude',
    #             'surface_names': ['farfield'],
    #             'per_surface': False,
    #             'average_over': 1,
    #             'retain_instantaneous_values': False
    #         }
    #     },
    #     'drag': {
    #         'drag-force': {
    #             'name': 'drag-force',
    #             'force_vector': [0.0, 1.0, 0.0],  # Opposite of freestream (-Y)
    #             'reference_frame': 'global',
    #             'zones': ['rocket_body'],
    #             'per_zone': False,
    #             'average_over': 1,
    #             'retain_instantaneous_values': False,
    #             'report_output_type': 'Drag Force'
    #         }
    #     },
    #     'lift': {
    #         'lift-force': {
    #             'name': 'lift-force',
    #             'report_type': 'force',
    #             'force_vector': [1.0, 0.0, 0.0],  # X is lift for -Y freestream
    #             'reference_frame': 'global',
    #             'zones': ['rocket_body'],
    #             'per_zone': False,
    #             'average_over': 1,
    #             'retain_instantaneous_values': False,
    #             'report_output_type': 'Lift Force'
    #         }
    #     },
    #     'moment': {
    #         'moment-z': {
    #             'name': 'moment-z',
    #             'report_type': 'moment',
    #             # 'moment_center': [0.0, 5.4862, 0.0],  # Change if needed
    #             # 'moment_axis': [0.0, 0.0, 1.0],  # About Z-axis
    #             # 'reference_frame': 'global',
    #             'zones': ['rocket_body'],
    #             # 'per_zone': False,
    #             # 'average_over': 1,
    #             # 'retain_instantaneous_values': False
    #         }
    #     }
    # })





    print("🧮 Running calculation...")
    session.settings.solution.run_calculation.iterate(iter_count=1)
    print("✅ Simulation complete.")


    print(f"Mach: {mach}")
    session.settings.solution.report_definitions.compute(report_defs=["vel_mag", "drag", "lift", "moment"])
    # print(f"✅ Drag @ Mach {mach} = {drag_val:.3f} N")

import pandas as pd
pd.DataFrame(results).to_csv("aero_sweep_results.csv", index=False)
