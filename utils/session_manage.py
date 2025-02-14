from typing import List

from app.const import Consts
from app.device import DeviceStructure, Layer
from app.light import Light
from app.nk import NK


def load_device_from_session(session, key="device_structure") -> DeviceStructure:
    layers = []
    for layer_data in session.get(key):
        if layer_data['nk']:
            layers.append(Layer(
                layer_data['layer_order'],
                layer_data['material'],
                layer_data['color'],
                layer_data['thickness'],
                layer_data['layer_type'],
                NK(layer_data['nk']['source'], layer_data['nk']['material'], layer_data['nk']['rfi_file_path'])
            ))
        else:
            layers.append(Layer(
                                layer_data['layer_order'],
                                layer_data['material'],
                                layer_data['color'],
                                layer_data['thickness'],
                                layer_data['layer_type'],
                                NK(None, None, None)
            ))
    device = DeviceStructure()
    device.set_layers(layers)
    return device


def load_light_from_session(session, key="light") -> Light:
    return Light(**session.get(key))


def save_light_to_session(session, light: Light, key="light"):
    session[key] = light.to_dict()


def save_device_to_session(session, device: DeviceStructure, key="device_structure"):
    session[key] = device.to_dict()


def save_nks_to_session(session, nks: List[NK], key="nk"):
    nks_json = []
    for nk in nks:
        nks_json.append(nk.to_dict())
    session[key] = nks_json


def load_nk_from_session(session, key="nk"):
    return session.get(key)
