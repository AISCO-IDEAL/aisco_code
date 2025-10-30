from dataclasses import dataclass, asdict
import cadquery as cq
from cadquery import Assembly, Color, Location
from ocp_vscode import *
from .seat import *
from .chassis import *
from .params import *
from .wheels import *
from utils.constraints import chassis_constraints, seat_constraints, wheels_constraints
from utils.randomize import *


class CarAssembly:
    def __init__(
        self,
        chassis_params: ChassisParams = ChassisParams(),
        seat_params: SeatParams = SeatParams(),
        wheel_params: WheelParams = WheelParams(),
    ):
        super().__init__()

        # -- creating models of the components --
        self.seat = Seat(params=seat_params, chassis_params=chassis_params)
        self.chassis = Chassis(
            chassis_params=chassis_params,
            seat_params=seat_params,
            wheel_params=wheel_params,
            seat=self.seat,
        )
        self.wheels = Wheels(params=wheel_params)

        # -- adding all components to the assembly --
        car_assy = Assembly(
            self.wheels.model, name="wheels", color=Color(wheel_params.color)
        )
        car_assy.add(
            self.chassis.model,
            loc=Location(0, 0, chassis_params.hover + chassis_params.height / 2),
            name="chassis",
            color=Color(chassis_params.color),
        )
        car_assy.add(
            self.seat.model,
            loc=Location(seat_params.position, 0, chassis_params.hover),
            name="seat",
            color=Color(seat_params.color),
        )
        self.model = car_assy

    def get_all_params(self):
        return {
            "chassis": asdict(self.chassis.params),
            "seat": asdict(self.seat.params),
            "wheels": asdict(self.wheels.params),
        }

    @staticmethod
    def randomized(
        random_magnitude_chassis: float = 0.0,
        random_magnitude_seat: float = 0.0,
        random_magnitude_wheels: float = 0.0,
        lock: list = [],
    ) -> "CarAssembly":
        original_ch_params = ChassisParams()
        original_s_params = SeatParams()
        original_w_params = WheelParams()

        ch_constraints = chassis_constraints(
            original_ch_params, original_s_params, original_w_params
        )
        s_constraints = seat_constraints(
            original_ch_params, original_s_params, original_w_params
        )
        w_constraints = wheels_constraints(
            original_ch_params, original_s_params, original_w_params
        )

        randomized_ch_params = randomize(
            original_ch_params,
            ch_constraints,
            random_magnitude=random_magnitude_chassis,
            lock=lock,
        )
        randomized_s_params = randomize(
            original_s_params,
            s_constraints,
            random_magnitude=random_magnitude_seat,
            lock=lock,
        )
        randomized_w_params = randomize(
            original_w_params,
            w_constraints,
            random_magnitude=random_magnitude_wheels,
            lock=lock,
        )

        return CarAssembly(
            chassis_params=randomized_ch_params,
            seat_params=randomized_s_params,
            wheel_params=randomized_w_params,
        )

    def export(self, path, mode="default"):
        """args:
            -path use .stl/.step for desired file type
            -mode
        for STL make sure to propeerly fuse the assembly
        change mode to 'fused' to make it one shape The following will attempt to create a single, fused shape while preserving the name and color information of each assembly object. The process of fusing the solid may cause performance issues in some cases, and is likely to alter the faces of the fused solids.
        **Important:** Use *export_compound* to export the outer shell!
        """
        self.model.export(path=path, mode=mode)

    def export_compound(self, path):
        """args:
            -path use .stl/.step for desired file type
            -mode
        Exports the outer shell Mesh as .stl
        """
        solid = self.model.toCompound().fuse()
        cq.exporters.export(solid, path)


# l = ['width', 'radius_top', 'plus_angle', 'minus_angle', 'width_hr', 'height_hr', 'Lr_l', 'Lr_h', 'radius', 'height']
# assy = CarAssembly.randomized(0.3,0.8,0.3,lock=l)
# assy = CarAssembly()
# print(assy.get_all_params())

# set_port(3939)
# show_object(assy.model)
