import StlAnalyzer
import matplotlib.pyplot as plt


def estimate_dynamic_cd(area: float, area_max: float, cd_max: float = 1.1, alpha: float = 1.0) -> float:
    """
    Estimate Cd at a given angle based on projected area ratio
    """
    if area_max == 0:
        return 0.0
    return cd_max * (area / area_max)**alpha


if __name__ == "__main__":

    # filepath = "C:/Users/emcof/PycharmProjects/ControlTheorySimulation/models/Aero/circle_disk.stl"
    filepath = 'C:/Users/emcof/PycharmProjects/ControlTheorySimulation/models/Aero/model_rocket.stl'
    mesh = StlAnalyzer.load_stl(filepath)
    angles, areas = StlAnalyzer.get_reference_area_vs_yaw(mesh, plane="XY", axis="y")
    area_max = max(areas)

    cds = [estimate_dynamic_cd(a, area_max, cd_max=1.0, alpha=1.2) for a in areas]

    for i in range(0, len(angles)):
        print(angles[i], cds[i])


    StlAnalyzer.plot_3x4_projections(mesh, plane="XY", axis="y")
    # plt.grid(True)
    # plt.show()
