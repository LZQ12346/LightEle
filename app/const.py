import math
import os


class Consts:
    LIGHT_SPEED = 2.99792458e8  # in meters per second (m/s)
    PLANCK = 6.62606957e-34  # in Joule seconds (J·s)
    ELEMENTARY_CHARGE = 1.60217657e-19  # in coulombs (C)
    VACUUM_PERMITTIVITY = 8.854187817e-12  # in farads per meter (F/m)
    NUMBER_OF_ORDERS = 31  # 计算精度
    PI = math.pi
    THETA0 = 4
    LAMBDA = 0.01
    N1 = 1
    N3 = 1
    LAYER_TYPES = ["正电极", "负电极", '吸收层', '电子传输层', '空穴传输层', '其他']
    VISIBLE_LIGHT_WAVE = (360, 830)
    SYSTEM_MATERIALS = list(i[:-4] for i in os.listdir(os.path.join("refractive_index_csv", "system")))

    NK_TYPE_CONST = "constant"
    NK_TYPE_SYSTEM = "system"
    NK_TYPE_CUSTOM = "custom"

    ART_EXCEL_HEADER = ['Wavelength (nm)', 'Transmission', 'Reflection', 'Absorption']
    ZLZ_GENERATION_RATE_HEADER = ['Position', 'Carrier generation rate']
    E_FIELD_COL_ONE_HEADER = "Position"


class FrontEndWarning:
    NO_JSON_BUT_FORM = "Do not support json, plz using form data."
    MISSING_JSON = "Missing JSON in request."
    INVALID_SOURCE = "Invalid refractive index source."
    FILE_ERROR = "No file uploaded or file format is not supported."
    FILE_EMPTY = "The uploaded file is empty."
    MATERIAL_NOT_FOUND = "This material does not offer in system."


class FileDir:
    CUSTOM_FRI_DIR = os.path.join("refractive_index_csv", "custom")
    TMP_DIR = os.path.join("tmp")
    IMAGES_DIR = os.path.join("output")
    EXCELS_DIR = os.path.join("output")
    DOWNLOAD_DIR = os.path.join("download")


class FileName:
    ART = "光学响应.xlsx"  # Absorption Reflection Transmission
    LIGHT_ELECTRICITY = "光生电流.xlsx"
    E_FIELD = "电场强度.xlsx"
    ZLZ_GENERATION_RATE = "载流子生成率.xlsx"
    ART_IMG = "光学响应.jpg"
    Gx_IMG = "载流子生成率.jpg"
    E_FIELD_IMG = "电场强度.jpg"
    E_FIELD_IMG_PREFIX = "电场强度"


class ImageInfo:
    # ART_TITLE = "光学响应图"
    # ART_Y_LABEL = "光学响应率"
    # ART_X_LABEL = "波长 (nm)"

    ART_TITLE = "Optical Response Graph"
    ART_Y_LABEL = "Optical Response Rate"
    ART_X_LABEL = "Wavelength (nm)"

    # ZLZ_GENERATION_RATE_TITLE = "载流子生成率图"
    # ZLZ_GENERATION_RATE_Y_LABEL = "载流子生成率"
    # ZLZ_GENERATION_RATE_X_LABEL = "位置 (nm)"

    ZLZ_GENERATION_RATE_TITLE = "Carrier Generation Rate Graph"
    ZLZ_GENERATION_RATE_Y_LABEL = "Carrier Generation Rate"
    ZLZ_GENERATION_RATE_X_LABEL = "Position (nm)"

    # E_FIELD_TITLE = "电场强度图"
    # E_FIELD_Y_LABEL = "电场强度"
    # E_FIELD_X_LABEL = "位置 (nm)"

    E_FIELD_TITLE = "Electric Field Strength Graph"
    E_FIELD_Y_LABEL = "Electric Field Strength"
    E_FIELD_X_LABEL = "Position (nm)"
