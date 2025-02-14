import os.path
from typing import List

import numpy as np

from .const import Consts, FrontEndWarning
from .nk import NK


class Layer:
    """The minimum unit of a device structure."""

    def __init__(self, layer_order: int, material: str, color, thickness: int, layer_type: str, nk: NK):
        """
        :param layer_order: start with 1
        :param material:
        :param color: in #RRGGBB format
        :param thickness: in nm
        :param layer_type:
        """
        self.layer_order_id = layer_order
        self.material = material
        self.color = color
        self.thickness = thickness
        self.layer_type = layer_type

        assert layer_type in Consts.LAYER_TYPES

        self.__nk = nk

    def set_nk(self, source, **kwargs):
        if source == Consts.NK_TYPE_CONST:     # 常数折射率
            self.__nk = NK(source, self.material)
            self.__nk.save_const_rfi(kwargs["rfi_real"], kwargs["rfi_imag"])  #  #  # 常熟
        elif source == Consts.NK_TYPE_SYSTEM:  # 使用系统提供的折射率文件
            std_material_name = kwargs['std_material_name']
            if os.path.exists(os.path.join("refractive_index_csv", "system", f"{std_material_name}.csv")):
                self.__nk = NK(source, std_material_name)
                self.__nk.set_system_rfi()
            else:
                raise FileNotFoundError(FrontEndWarning.MATERIAL_NOT_FOUND)
        elif source == Consts.NK_TYPE_CUSTOM:  # 用户自定义折射率文件，需要上传
            self.__nk = NK(source, self.material)
            self.__nk.save_custom_rfi(kwargs["rfi_file"], kwargs["user"])
        return self.__nk.get_rfi_file_path()

    def get_nk(self):
        if not self.__nk:
            raise ValueError("Not set refractive index data yet.")
        return self.__nk

    def __str__(self):
        return f"{self.layer_order_id}\t{self.material}\t{self.color}\t{self.thickness}\t{self.layer_type}\t{self.__nk.get_source()}\t      {self.__nk.get_rfi_file_path()}"


    def to_dict(self):
        return {
            "layer_order": self.layer_order_id,
            "material": self.material,
            "color": self.color,
            "thickness": self.thickness,
            "layer_type": self.layer_type,
            "nk": self.__nk.to_dict() if self.__nk else None
        }


class DeviceStructure:
    """
    器件结构
    """

    def __init__(self):
        self.__layers: List[Layer] = []
        self.__layer_num = len(self.__layers)
        self.mesh_density = 1

    def get_layer_num(self) -> int:
        return self.__layer_num

    def is_empty_device(self) -> bool:
        return self.__layer_num == 0

    def get_layer(self, layer_order: int) -> Layer:
        return self.__layers[layer_order - 1]

    def set_layers(self, layers: list):
        self.__layers = layers
        self.__layer_num = len(layers)

    def get_all_layers(self) -> list:
        return self.__layers

    def get_total_thickness_in_meter(self) -> float:
        return sum([int(layer.thickness) for layer in self.__layers]) * 10 ** -9

    def get_total_thickness_in_nm(self) -> float:
        return sum([layer.thickness for layer in self.__layers])

    def get_layer_thicknesses_in_meter(self) -> np.ndarray:
        return np.array([int(layer.thickness) for layer in self.__layers]) * 10 ** -9

    def add_layer(self, layer: Layer):
        self.__layers.append(layer)
        self.__layer_num += 1

    def remove_layer(self, layer_order: int):
        if self.__layer_num < 0:
            raise ValueError('No layer to remove.')
        try:
            del self.__layers[layer_order - 1]
            self.__layer_num -= 1
        except IndexError:
            raise ValueError('The layer does not exist.')

    def move_up(self, layer_order: int):
        if self.__layer_num <= 1:
            raise ValueError('There is only one or no layer, cannot move up.')
        if layer_order == 1:
            raise ValueError('The first layer cannot be moved up.')
        self.__layers[layer_order - 1], self.__layers[layer_order - 2] = self.__layers[layer_order - 2], self.__layers[
            layer_order - 1]

    def move_down(self, layer_order: int):
        if self.__layer_num <= 1:
            raise ValueError('There is only one or no layer, cannot move down.')
        if layer_order == self.__layer_num:
            raise ValueError('The last layer cannot be moved down.')
        self.__layers[layer_order], self.__layers[layer_order - 1] = self.__layers[layer_order - 1], self.__layers[
            layer_order]

    def to_dict(self):
        return [layer.to_dict() for layer in self.__layers]

    def get_absorption_layer_order_id(self):
        for i, layer in enumerate(self.__layers):
            if layer.layer_type == '吸收层':
                return i + 1
        return -1

    def __str__(self):
        print("层号\t材料\t颜色\t厚度\t类型\t折射率来源\t\t文件路径")
        for layer in self.__layers:
            print(layer)
        return ""

    def __repr__(self):
        return self.__layers
