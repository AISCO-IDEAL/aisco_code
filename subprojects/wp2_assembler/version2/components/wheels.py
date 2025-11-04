from dataclasses import dataclass  # for defining parameter classes
from utils.reg_imp import get_regulation  # import regulation data


# --- Wheels class: ---
@dataclass
class Wheels:
    # import from regulations wheel dimensions
    wheel_count: int = 3
    wheel_radius: float = 0.3

    def get_wheel_info(self) -> str:
        return f"This vehicle has {self.wheel_count} {self.wheel_type} wheels."
