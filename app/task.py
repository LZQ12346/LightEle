import os
from utils import now
from .device import DeviceStructure
from .light import Light
from .rcwa import calculate


class Task:
    def __init__(self, user_id: str, device: DeviceStructure, light: Light):
        self.user_id = user_id
        self.device = device
        self.light = light
        self.result_files: tuple = ()
        self.run_time = now()

    def run(self) -> None:
        self.result_files = calculate(self.device, self.light, self.user_id, self.run_time)

    def check_result_files(self) -> bool:
        excels = self.result_files[0]
        images = self.result_files[1]
        zip_file = self.result_files[2]
        return all(os.path.exists(i) for i in excels + images + [zip_file])

    def save_task(self):
        save_dict = {
            'user_id': self.user_id,
            'device': self.device,
            'light': self.light,
            'result': self.result
        }
        save_dir = os.path.join("tasks")
