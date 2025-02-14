import numpy as np


class Light:
    def __init__(self, min_lam, max_lam, step_length, mesh_grid, spectrum):
        self.__min_lam = min_lam
        self.__max_lam = max_lam
        self.__step_length = step_length
        self.__mesh_grid = mesh_grid
        self.__spectrum = spectrum  # 1: solar spectrum AM1.5

    def get_min_lam(self):
        return self.__min_lam

    def get_max_lam(self):
        return self.__max_lam

    def get_step(self):
        return self.__step_length

    def get_mesh_grid(self):
        return self.__mesh_grid

    def set_min_lam(self, min_lam):
        self.__min_lam = min_lam

    def set_max_lam(self, max_lam):
        self.__max_lam = max_lam

    def set_step_len(self, step_length):
        self.__step_length = step_length

    def set_mesh_grid(self, mesh_grid):
        self.__mesh_grid = mesh_grid

    def set_spectrum(self, spectrum):
        self.__spectrum = "AM1.5" if spectrum == 1 else "None"

    def get_lams_in_list(self) -> list:
        return np.around(np.arange(self.__min_lam, self.__max_lam, self.__step_length), decimals=4).tolist()

    def get_lams_in_ndarray(self) -> np.ndarray:
        return np.around(np.arange(self.__min_lam, self.__max_lam, self.__step_length), decimals=4)

    def check_valid(self):
        return self.__max_lam > self.__min_lam > 0 and 0 < self.__step_length < self.__max_lam - self.__min_lam

    def __str__(self):
        return f"{self.__min_lam}\t{self.__max_lam}\t{self.__step_length}\t{self.__spectrum}"

    def to_dict(self):
        return {
            'min_lam': self.__min_lam,
            'max_lam': self.__max_lam,
            'step_length': self.__step_length,
            'mesh_grid': self.__mesh_grid,
            'spectrum': "AM1.5" if self.__spectrum == 1 else "other"
        }
