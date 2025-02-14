import os
import pandas as pd


from utils import check_nk_csv_valid
from .const import Consts, FileDir


class NK:
    """折射率类"""

    def __init__(self, source: str, material: str, rfi_file_path=None):
        self.__source = source
        self.__material = material
        self.__rfi_file_path = rfi_file_path  # Should be set later

    def get_source(self):
        return self.__source

    def get_rfi_file_path(self):
        return self.__rfi_file_path

    def save_const_rfi(self, rfi_real: float, rfi_imag: float):
        """
        For rfi-source: const
        :return: bool
        """
        try:
            const_dir = os.path.join("refractive_index_csv", "const")
            if not os.path.exists(const_dir):
                os.makedirs(const_dir)
            rfi_save_path = os.path.join(const_dir, f"{self.__material}.csv")
            df = pd.DataFrame({"Wavelength (nm)": [Consts.VISIBLE_LIGHT_WAVE[0], Consts.VISIBLE_LIGHT_WAVE[1]],
                               f"{self.__material}_n": [rfi_real] * 2,
                               f"{self.__material}_k": [rfi_imag] * 2})
            df.to_csv(rfi_save_path, index=False, header=False)
            self.__rfi_file_path = rfi_save_path
        except FileNotFoundError as fnfe:
            return False
        except Exception as e:
            return False

    def save_custom_rfi(self, tmp_file_path, user):
        if check_nk_csv_valid(tmp_file_path):
            rfi_save_dir = os.path.join("refractive_index_csv", "custom", str(user.get_userId()))
            if not os.path.exists(rfi_save_dir):
                os.makedirs(rfi_save_dir)
            rfi_save_path = os.path.join(rfi_save_dir, f"{self.__material}.csv")
            pd.read_csv(tmp_file_path, header=None).to_csv(rfi_save_path, index=False, header=False)  # Remove header
            self.__rfi_file_path = rfi_save_path
            return True
        return False

    def set_system_rfi(self, ):
        self.__rfi_file_path = os.path.join("refractive_index_csv", "system", f"{self.__material}.csv")

    def __str__(self):
        return f"Material: {self.__material}\tSource: {self.__source}\tFile path: {self.__rfi_file_path}"

    def to_dict(self):
        return {
            "source": self.__source,
            "material": self.__material,
            "rfi_file_path": self.__rfi_file_path
        }