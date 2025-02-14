import bisect
import math
import os

import numpy as np
import pandas as pd
import seaborn as sns
import xlrd.biffh
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

import utils as u
from .const import FileDir, Consts, FileName, ImageInfo
from .device import DeviceStructure, Layer
from .light import Light


def line(x1, x2, y1, y2, x):
    k = (y1 - y2) / (x1 - x2)
    b = y2 - k * x2
    y = k * x + b
    return y


def refractive_index(lam: int, layer: Layer):
    """
    查找某（单）个材料在某个波长下的折射率，并对折射率做延拓
    :param layer:
    :param lam: 波长
    :return:
    """

    df = pd.read_csv(layer.get_nk().get_rfi_file_path(), header=None)
    lstWave, lstn, lstk = list(df.iloc[:, 0]), list(df.iloc[:, 1]), list(df.iloc[:, 2])
    if lam >= lstWave[-1]:
        strn = line(lstWave[-1], lstWave[-2], lstn[-1], lstn[-2], lam)
        strk = line(lstWave[-1], lstWave[-2], lstk[-1], lstk[-2], lam)
    elif lam < lstWave[0]:
        strn = line(lstWave[0], lstWave[1], lstn[0], lstn[1], lam)
        strk = line(lstWave[0], lstWave[1], lstk[0], lstk[1], lam)
    else:
        strn = line(lstWave[bisect.bisect(lstWave, lam) - 1], lstWave[bisect.bisect(lstWave, lam)],
                    lstn[bisect.bisect(lstWave, lam) - 1], lstn[bisect.bisect(lstWave, lam)], lam)
        strk = line(lstWave[bisect.bisect(lstWave, lam) - 1], lstWave[bisect.bisect(lstWave, lam)],
                    lstk[bisect.bisect(lstWave, lam) - 1], lstk[bisect.bisect(lstWave, lam)], lam)
    n = strn - strk * 1j
    return n


def RCWA(lam: int, device_struct: DeviceStructure, mesh_density=1) -> tuple:
    """
    计算单个波长下，每一层的吸收和吸收矩阵,反射和透射
    :param device_struct: 器件结构
    :param lam: 照射波长
    :param mesh_density: 网格密度
    :return: (list, np.float64, np.float64, np.ndarray, total_thickness)
    """
    layer_num = device_struct.get_layer_num()
    total_thickness = device_struct.get_total_thickness_in_meter()
    layer_thickness = device_struct.get_layer_thicknesses_in_meter()

    nrd = np.array(np.zeros((1, layer_num)), dtype=complex)
    for i in range(device_struct.get_layer_num()):
        nrd[0, i] = refractive_index(lam, device_struct.get_layer(i + 1))
    nrd = nrd[0]

    ngr = np.array(np.zeros((1, layer_num)), dtype=complex)
    for i in range(device_struct.get_layer_num()):
        ngr[0, i] = refractive_index(lam, device_struct.get_layer(i + 1))
    ngr = ngr[0]

    # 网格密度
    if mesh_density == 1:
        resolution_x = 30
        if total_thickness >= 10 * 10 ** -6:
            if lam >= 500:
                resolution_z = 15000
            else:
                resolution_z = 100000
        elif 5 * 10 ** -6 <= total_thickness < 1 * 10 ** -5:
            resolution_z = 10000
        else:
            resolution_z = 500
    elif mesh_density == 2:
        resolution_x = 40
        if total_thickness >= 10 * 10 ** -6:
            if lam >= 500:
                resolution_z = 15000
            else:
                resolution_z = 150000
        elif 5 * 10 ** -6 <= total_thickness < 1 * 10 ** -5:
            resolution_z = 15000
        else:
            resolution_z = 15000
    elif mesh_density == 3:
        resolution_x = 50
        if total_thickness >= 10 * 10 ** -6:
            if lam >= 500:
                resolution_z = 20000
            else:
                resolution_z = 200000
        elif 5 * 10 ** -6 <= total_thickness < 1 * 10 ** -5:
            resolution_z = 20000
        else:
            resolution_z = 20000
    else:
        raise ValueError('The mesh density should be 1, 2 or 3.')

    order_max = int((Consts.NUMBER_OF_ORDERS - 1) / 2)
    n = np.array(range(-order_max, order_max + 1)).T
    theta = Consts.THETA0 * Consts.PI / 180
    k_0 = 2 * Consts.PI / (lam * 10 ** -9)
    phi = 0
    K = 2 * Consts.PI / (Consts.LAMBDA * 10 ** -6)
    I = np.eye(Consts.NUMBER_OF_ORDERS)
    k_x = k_0 * Consts.N1 * math.sin(theta) * math.cos(phi) + n * K
    k_x_2 = k_x ** 2
    K_x = np.diag(k_x / k_0)
    K_x_2 = K_x ** 2
    k_1_z = np.array(np.zeros((1, Consts.NUMBER_OF_ORDERS)), dtype=complex)
    for m in range(0, Consts.NUMBER_OF_ORDERS):
        if np.sqrt(k_x_2[m]) < k_0 * Consts.N1:
            k_1_z[0, m] = math.sqrt((k_0 * Consts.N1) ** 2 - k_x_2[m])
        else:
            k_1_z[0, m] = -1j * math.sqrt((k_x_2[m]) - (k_0 * Consts.N1) ** 2)

    k_3_z = np.array(np.zeros((1, Consts.NUMBER_OF_ORDERS)), dtype=complex)
    for m in range(0, Consts.NUMBER_OF_ORDERS):
        if np.sqrt(k_x_2[m]) < k_0 * Consts.N3:
            k_3_z[0, m] = math.sqrt((k_0 * Consts.N3) ** 2 - k_x_2[m])
        else:
            k_3_z[0, m] = -1j * math.sqrt((k_x_2[m]) - (k_0 * Consts.N3) ** 2)
    k_1_z_n = np.array(np.zeros((1, Consts.NUMBER_OF_ORDERS)), dtype=complex)
    k_3_z_n = np.array(np.zeros((1, Consts.NUMBER_OF_ORDERS)), dtype=complex)

    for i in range(0, Consts.NUMBER_OF_ORDERS):
        if np.imag(k_1_z[0, i]) == 0:
            k_1_z_n[0, i] = -k_1_z[0, i]
        else:
            k_1_z_n[0, i] = k_1_z[0, i]
        if np.imag(k_3_z[0, i]) == 0:
            k_3_z_n[0, i] = -k_3_z[0, i]
        else:
            k_3_z_n[0, i] = k_3_z[0, i]

    new_T = I
    k_1_z_k_0 = k_1_z / k_0
    Y1 = np.diag(k_1_z_k_0[0])

    k_3_z_k_0 = k_3_z / k_0
    Y3 = np.diag(k_3_z_k_0[0])

    incident = np.zeros((1, 2 * Consts.NUMBER_OF_ORDERS), dtype=complex)
    incident[0, order_max] = 1
    incident[0, Consts.NUMBER_OF_ORDERS + order_max] = 1j * Consts.N1 * math.cos(theta)
    f_g = np.vstack((I, 1j * Y3))
    cw = np.zeros([1, Consts.NUMBER_OF_ORDERS])
    epr_h = np.zeros((1, 2 * Consts.NUMBER_OF_ORDERS - 1), dtype=complex)
    E_l = np.zeros((Consts.NUMBER_OF_ORDERS, Consts.NUMBER_OF_ORDERS, layer_num), dtype=complex)
    W = np.zeros((Consts.NUMBER_OF_ORDERS, Consts.NUMBER_OF_ORDERS, layer_num), dtype=complex)
    Q = np.zeros((Consts.NUMBER_OF_ORDERS, Consts.NUMBER_OF_ORDERS, layer_num), dtype=complex)
    V = np.zeros((Consts.NUMBER_OF_ORDERS, Consts.NUMBER_OF_ORDERS, layer_num), dtype=complex)
    X = np.zeros((Consts.NUMBER_OF_ORDERS, Consts.NUMBER_OF_ORDERS, layer_num), dtype=complex)
    a_1_X = np.zeros((Consts.NUMBER_OF_ORDERS, Consts.NUMBER_OF_ORDERS, layer_num), dtype=complex)
    b_a_1_X = np.zeros((Consts.NUMBER_OF_ORDERS, Consts.NUMBER_OF_ORDERS, layer_num), dtype=complex)

    for k in range(1, Consts.NUMBER_OF_ORDERS + 1):
        cw[0, k - 1] = math.floor(k / 2) * ((-1) ** k)
    for l in range(layer_num - 1, -1, -1):
        epsr = nrd[l] ** 2
        four = np.array(range(1 - Consts.NUMBER_OF_ORDERS, Consts.NUMBER_OF_ORDERS))
        epr_h[0, Consts.NUMBER_OF_ORDERS - 1] = epsr

        for ii in range(0, Consts.NUMBER_OF_ORDERS):
            for p in range(0, Consts.NUMBER_OF_ORDERS):
                E_l[ii, p, l] = epr_h[0, int(cw[0, ii] - cw[0, p] - four[1] + 1)]

        eigenvalue_equation = K_x_2 - E_l[:, :, l]
        [Qq, WW] = np.linalg.eig(eigenvalue_equation)
        ind = np.argsort(Qq)
        Qq = Qq[ind]
        WW = WW[:, ind]
        del eigenvalue_equation
        W[:, :, l] = WW
        Q1 = np.sqrt(Qq)
        Q[:, :, l] = np.diag(Q1)
        V[:, :, l] = np.matmul(W[:, :, l], Q[:, :, l])
        S1 = np.append(W[:, :, l], W[:, :, l], axis=1)
        S2 = np.append(V[:, :, l], -V[:, :, l], axis=1)
        auxiliary_matrix = np.append(S1, S2, axis=0)
        X[:, :, l] = np.diag(np.exp(-k_0 * np.diag(Q[:, :, l]) * layer_thickness[l]))
        a_b = np.matmul(np.linalg.inv(auxiliary_matrix), f_g)
        a = a_b[0:Consts.NUMBER_OF_ORDERS, 0:Consts.NUMBER_OF_ORDERS]
        b = a_b[Consts.NUMBER_OF_ORDERS:2 * Consts.NUMBER_OF_ORDERS + 1, 0:Consts.NUMBER_OF_ORDERS]
        a_1_X[:, :, l] = np.matmul(np.linalg.inv(a), X[:, :, l])
        b_a_1_X[:, :, l] = np.matmul(b, a_1_X[:, :, l])
        X_b_a_1_X = np.matmul(X[:, :, l], b_a_1_X[:, :, l])
        f_g = np.append(np.matmul(W[:, :, l], (np.eye(Consts.NUMBER_OF_ORDERS) + X_b_a_1_X)),
                        np.matmul(V[:, :, l], (np.eye(Consts.NUMBER_OF_ORDERS) - X_b_a_1_X)), axis=0)
        new_T = np.matmul(new_T, a_1_X[:, :, l])

    transfer = np.hstack((np.vstack((-I, 1j * Y1)), f_g))
    R_T1 = np.matmul(np.linalg.inv(transfer), incident.T)
    R = R_T1[0:Consts.NUMBER_OF_ORDERS, 0]
    T1 = R_T1[Consts.NUMBER_OF_ORDERS:2 * Consts.NUMBER_OF_ORDERS + 1, 0]
    T = np.matmul(new_T, T1)
    del transfer
    D_R = np.real((R * np.conj(R) * np.real(k_1_z_k_0 / (Consts.N1 * math.cos(theta)))))
    D_T = np.real((T * np.conj(T) * np.real(k_3_z_k_0 / (Consts.N1 * math.cos(theta)))))
    ret_t = sum(sum(D_T))
    ret_r = sum(sum(D_R))

    perioda = Consts.LAMBDA * 10 ** -6
    epsilon = Consts.VACUUM_PERMITTIVITY
    number_of_plotted_period = 1
    thickness = total_thickness
    x_min = 0
    x_max = number_of_plotted_period * perioda
    x_res_ = perioda / resolution_x
    z_res_ = thickness / resolution_z
    T_l = T1
    D_l_pred = 0
    x = np.arange(x_min, x_max + x_res_, x_res_)
    absorption_in_layer_thickness = []
    exp_xx = np.zeros((Consts.NUMBER_OF_ORDERS, len(x)), dtype=complex)
    epsilon_matrix_local = np.empty([len(x), 0])
    field_E_2 = np.empty([len(x), 0])
    for i in range(0, len(x)):
        exp_xx[:, i] = np.exp(-1j * k_x * x[i])
    for l in range(0, layer_num):
        epsg = ngr[l] ** 2
        D_l = D_l_pred + layer_thickness[l]
        z = np.array(np.arange((D_l_pred + z_res_), D_l, z_res_))
        absorption_in_layer_thickness.append(np.size(z))
        z_plus = z - D_l_pred
        z_minus = z - D_l
        E_y = np.zeros((np.size(x), np.size(z)), dtype=complex)
        c_plus_minus = np.matmul(np.vstack((I, b_a_1_X[:, :, l])), T_l)
        c_plus = c_plus_minus[0:Consts.NUMBER_OF_ORDERS]
        c_minus = c_plus_minus[Consts.NUMBER_OF_ORDERS:np.size(c_plus_minus) + 1]
        epsilon_matrix_local_ = np.empty([len(x), len(z)], dtype=complex)
        for z_l in range(0, len(z)):
            c_plus_exp = c_plus * np.exp(-k_0 * np.diag(Q[:, :, l]) * z_plus[z_l])
            c_minus_exp = c_minus * np.exp(k_0 * np.diag(Q[:, :, l]) * z_minus[z_l])
            S_harmonics = np.matmul(W[:, :, l], (c_plus_exp + c_minus_exp))
            E_y[:, z_l] = np.matmul(S_harmonics.T, exp_xx)
            for x_l in range(0, len(x)):
                epsilon_matrix_local_[x_l, z_l] = epsilon * epsg
        epsilon_matrix_local = np.hstack((epsilon_matrix_local, epsilon_matrix_local_))
        T_l = np.matmul(a_1_X[:, :, l], T_l)
        D_l_pred = D_l
        field_E = E_y
        field_E_2 = np.hstack((field_E_2, field_E))
        del epsilon_matrix_local_
    component_E_y = field_E_2
    intensity_E = (abs(component_E_y)) ** 2
    local_absorption_matrix = -(
            (k_0 ** 2) / k_1_z[0, int((Consts.NUMBER_OF_ORDERS + 1) / 2 - 1)] * z_res_ / resolution_x) * (
                                      intensity_E * np.imag(epsilon_matrix_local / epsilon))
    z_min = 0
    z_max = absorption_in_layer_thickness[0]
    layer_absorption = []
    for i in range(0, layer_num):
        layer_absorption.append(sum(sum(local_absorption_matrix[:, z_min:z_max])))
        if i < layer_num - 1:
            z_min = z_min + absorption_in_layer_thickness[i];
            z_max = z_max + absorption_in_layer_thickness[i + 1]
    return layer_absorption, ret_r, ret_t, local_absorption_matrix, total_thickness, abs(component_E_y)


def jph(light: Light, Ab: np.ndarray, ab: np.ndarray, total_thickness) -> tuple:
    """
    光生电流密度j和载流子生成率gxsumlam，每一层材料光生电流值
    :param total_thickness: from the RCWA function
    :param light: Light object
    :param Ab: np.ndarray
    :param ab: np.ndarray
    :return: (np.ndarray（）, np.ndarray)
    """
    dfAM15 = pd.read_excel(os.path.join("spectrum", "AM15Gtest.xls"))
    dicAM15 = dfAM15.to_dict()
    dicWave = dicAM15['Wvlgth nm']
    lstWave = list(dicWave.values())
    dicy = dicAM15['AM']
    lsty = list(dicy.values())
    am15_interp = interp1d(lstWave, lsty, 'linear')
    am15_int_y = []

    for idx in range(light.get_min_lam(), light.get_max_lam(), light.get_step()):
        if idx >= lstWave[-1]:
            am15_int_yy = am15_interp(lstWave[-1])
        elif idx < lstWave[0]:
            am15_int_yy = am15_interp(lstWave[0])
        else:
            am15_int_yy = am15_interp(idx)
        am15_int_y.append(am15_int_yy)
    am15_int_y = np.array(am15_int_y)

    Gx = Ab.T * am15_int_y * 1e-14 / (total_thickness / np.shape(Ab)[1])
    gxsumlam = Gx.sum(axis=1)

    j = ab.T * am15_int_y * light.get_step() * Consts.ELEMENTARY_CHARGE * 1e-4 * 1e-13
    j = j.sum(axis=1) * 1e7
    return j, gxsumlam


def draw_art(df: pd.DataFrame, save_dir, filename):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=Consts.ART_EXCEL_HEADER[0], y=Consts.ART_EXCEL_HEADER[1], data=df, label=Consts.ART_EXCEL_HEADER[1])
    sns.lineplot(x=Consts.ART_EXCEL_HEADER[0], y=Consts.ART_EXCEL_HEADER[2], data=df, label=Consts.ART_EXCEL_HEADER[2])
    sns.lineplot(x=Consts.ART_EXCEL_HEADER[0], y=Consts.ART_EXCEL_HEADER[3], data=df, label=Consts.ART_EXCEL_HEADER[3])

    plt.legend(title='Curves')

    plt.xlabel(ImageInfo.ART_X_LABEL)
    plt.ylabel(ImageInfo.ART_Y_LABEL)
    plt.title(ImageInfo.ART_TITLE)

    plt.savefig(os.path.join(save_dir, filename), format='jpg')
    plt.close()

def draw_Gx(df: pd.DataFrame, save_dir, filename):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=Consts.ZLZ_GENERATION_RATE_HEADER[0], y=Consts.ZLZ_GENERATION_RATE_HEADER[1],
                 data=df, label=Consts.ZLZ_GENERATION_RATE_HEADER[1])

    plt.legend(title='Curves')

    plt.xlabel(ImageInfo.ZLZ_GENERATION_RATE_X_LABEL)
    plt.ylabel(ImageInfo.ZLZ_GENERATION_RATE_Y_LABEL)
    plt.title(ImageInfo.ZLZ_GENERATION_RATE_TITLE)

    plt.savefig(os.path.join(save_dir, filename), format='jpg')
    plt.close()

def draw_E_field(light: Light, df: pd.DataFrame, save_dir, filename):
    for i, index in enumerate(range(light.get_min_lam(),light.get_max_lam(),light.get_step())):
        indexx = str(index)
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        E_y=df['%snm' %indexx]
        plt.plot(df['Position'],E_y)
        plt.xlabel(ImageInfo.E_FIELD_X_LABEL)
        plt.ylabel(ImageInfo.E_FIELD_Y_LABEL)
        plt.title(f"{ImageInfo.E_FIELD_TITLE} {index}nm")
        plt.savefig(os.path.join(save_dir, f"{filename.split('.')[0]}_{i}.{filename.split('.')[1]}"), format='jpg')
        plt.close()

def save_dataframe_to_excel(target_df, save_dir: str, filename: str):
    df = pd.DataFrame(target_df)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_excel(os.path.join(save_dir, filename), index=False)


def calculate(device: DeviceStructure, light: Light, user_id: str, now_time: str) -> tuple:
    """完成最终计算，返回生成的文件路径列表(list[excel files], list[images], zipfile)"""
    arr_transmission = np.zeros((np.size(light.get_lams_in_ndarray()), 1), dtype=complex)  # 透射
    arr_reflection = np.zeros((np.size(light.get_lams_in_ndarray()), 1), dtype=complex)    # 反射
    arr_absorption = np.zeros((np.size(light.get_lams_in_ndarray()), 1), dtype=complex)    # 吸收
    Ab, ab = [], []
    total_thickness = None  # value will be updated in the RCWA function, and used in the jph function

    ele_fld_header = [Consts.E_FIELD_COL_ONE_HEADER]
    ele_fld_data = []
    for index in range(0, np.size(light.get_lams_in_ndarray())):
        try:
            layer_absorption, ret_r, ret_t, local_absorption_matrix, total_thickness, E = RCWA(
                light.get_lams_in_list()[index],
                device,
                mesh_density=light.get_mesh_grid())  # TODO Warning: 此函数不停IO
            arr_transmission[index, 0] = ret_t
            arr_reflection[index, 0] = ret_r
            arr_absorption[index, 0] = layer_absorption[device.get_absorption_layer_order_id() - 1]
            Ab.append(local_absorption_matrix.sum(axis=0))
            ab.append(layer_absorption)

            ele_fld_header.append(f"{light.get_lams_in_list()[index]}nm")
            ele_fld_data.append(E[0])
        except xlrd.biffh.XLRDError as e:
            print(f"Error: {e}，文件被加密")
            continue
        except FileNotFoundError as e:
            print(f"Error: {e}，文件不存在")
            continue

    J, Gxsumlam = jph(light, np.array(Ab), np.array(ab), total_thickness)

    position = np.array(range(0, len(Gxsumlam))) * total_thickness / len(Gxsumlam)
    df_position = pd.DataFrame(position.real)  # 位置坐标

    ele_fld_data.insert(0, position.real)

    df_E_field = pd.DataFrame({header: pd.Series(data) for header, data in zip(ele_fld_header, ele_fld_data)})

    df_zlz_generation_rate = pd.concat([df_position, pd.DataFrame(Gxsumlam.real)], axis=1)
    df_zlz_generation_rate.columns = Consts.ZLZ_GENERATION_RATE_HEADER

    df_art = pd.concat([pd.DataFrame(light.get_lams_in_ndarray()),
                             pd.DataFrame(arr_transmission.real),
                             pd.DataFrame(arr_reflection.real),
                             pd.DataFrame(arr_absorption.real)],
                       axis=1)  # Absorption Reflection Transmission (art)
    df_art.columns = Consts.ART_EXCEL_HEADER

    # 保存的文件均须在各自的用户目录下，且目录下还有带时间戳的文件夹。类似于：
    # 13/2024-12-12-12-12-12/*.xls
    # 保存数据到excel
    excel_save_dir = os.path.join(FileDir.EXCELS_DIR, user_id, now_time)
    save_dataframe_to_excel(df_E_field, excel_save_dir, FileName.E_FIELD)                          # 电场强度
    save_dataframe_to_excel(J, excel_save_dir, FileName.LIGHT_ELECTRICITY)                         # 光生电流
    save_dataframe_to_excel(df_zlz_generation_rate, excel_save_dir, FileName.ZLZ_GENERATION_RATE)  # 载流子生成率
    save_dataframe_to_excel(df_art, excel_save_dir, FileName.ART)                                  # 光学响应

    # 绘图
    jpg_save_dir = os.path.join(FileDir.IMAGES_DIR, user_id, now_time)
    draw_art(df_art, jpg_save_dir, FileName.ART_IMG)
    draw_Gx(df_zlz_generation_rate, jpg_save_dir, FileName.Gx_IMG)
    draw_E_field(light, df_E_field, jpg_save_dir, FileName.E_FIELD_IMG)

    # 生成zip文件
    u.zip_files(source_dir=os.path.join(FileDir.EXCELS_DIR, user_id, now_time),
                output_zip_dir=os.path.join(FileDir.DOWNLOAD_DIR, user_id), filename=f"output_{now_time}.zip")
    return ([
        os.path.join(excel_save_dir, FileName.E_FIELD),
        os.path.join(excel_save_dir, FileName.LIGHT_ELECTRICITY),
        os.path.join(excel_save_dir, FileName.ZLZ_GENERATION_RATE),
        os.path.join(excel_save_dir, FileName.ART),
    ], [
        os.path.join(jpg_save_dir, FileName.ART_IMG),
        os.path.join(jpg_save_dir, FileName.Gx_IMG)
    ],
    os.path.join(FileDir.DOWNLOAD_DIR, user_id, f"output_{now_time}.zip"))