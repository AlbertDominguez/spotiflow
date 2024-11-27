import torch
import abc

class BaseTestTimeAugmentation(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def apply(self, arr: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def revert(self, arr: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

class Rotation90TestTimeAugmentation(BaseTestTimeAugmentation):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, arr: torch.Tensor, angle: int) -> torch.Tensor:
        assert angle in (0, 90, 180, 270), "Angle must be 90, 180, or 270."
        if angle == 0:
            return arr
        return torch.rot90(arr, k=angle//90, dims=(-2, -1))

    def revert(self, arr: torch.Tensor, angle: int) -> torch.Tensor:
        assert angle in (0, 90, 180, 270), "Angle must be 90, 180, or 270."
        if angle == 0:
            return arr
        return torch.rot90(arr, k=4-angle//90, dims=(-2, -1))

if __name__ == "__main__":
    arr = torch.rand(64,64,64)
    tta = Rotation90TestTimeAugmentation()
    angles = [0, 90, 180, 270]
    for angle in angles:
        arr_rotated = tta.apply(arr, angle=angle)
        arr_reverted = tta.revert(arr_rotated, angle=angle)
        print(torch.allclose(arr, arr_reverted))
