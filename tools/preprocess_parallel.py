import matplotlib
import matplotlib.pyplot as plt

import os
import sys
import time
import random as rdn
import pdb

import h5py
import yaml
import json
import copy
import collections
import argparse

import math
import numpy as np
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
import multiprocessing
from collections import OrderedDict

# from yaml import CLoader as Loader, CDumper as Dumper
# from yaml.representer import SafeRepresenter
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))


# Dumper.add_representer(OrderedDict, dict_representer)
# Loader.add_constructor(_mapping_tag, dict_constructor)
# Dumper.add_representer(str, SafeRepresenter.represent_str)

# custom libs
import _init_paths
from lib.transformations import euler_matrix, quaternion_matrix
from lib.vis_utils import plot3d_pts
from lib.data_utils import collect_file, split_dataset
from global_info import global_info

from lib.data_utils import get_model_pts, get_urdf, get_urdf_mobility


class PoseDataset:
    def __init__(
        self,
        root_dir,
        item,
        num_points=1024,
        objs=[],
        add_noise=False,
        noise_trans=0,
        mode="train",
        refine=False,
        selected_list=None,
        is_debug=False,
    ):
        """
        num is the number of points chosen feeding into PointNet
        """
        self.is_debug = is_debug
        self.mode = mode
        self.max_lnk = 10
        self.root_dir = root_dir
        self.dataset_render = root_dir + "/render"
        # models_dir no use
        self.models_dir = root_dir + "/objects"
        self.objnamelist = os.listdir(self.dataset_render)
        self.mode = mode
        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_instance = []
        # obj/instance/arti

        self.list_status = []
        self.list_rank = []
        self.meta_dict = {}
        self.urdf_dict = {}
        self.pt_dict = {}
        self.noise_trans = noise_trans
        self.refine = refine

        ins_count = 0
        if self.is_debug:
            from mpl_toolkits.mplot3d import Axes3D
        obj_category = item
        instances_per_obj = os.listdir(self.dataset_render + "/" + obj_category)
        instances_per_obj.sort()
        meta_dict_obj = {}
        urdf_dict_obj = {}
        for ins in instances_per_obj:
            # pdb.set_trace()
            if selected_list is not None and ins not in selected_list:
                continue
            base_path = self.dataset_render + "/" + obj_category + "/" + ins
            print(base_path)
            meta = {}
            urdf_ins = {}  # link, joint
            # ********* add pts ************ #
            for art_index in os.listdir(base_path):
                sub_dir0 = base_path + "/" + art_index
                input_file = open(sub_dir0 + "/all.txt")
                while 1:
                    ins_count += 1
                    input_line = input_file.readline()
                    if not input_line:
                        break
                    if input_line[-1:] == "\n":
                        input_line = input_line[:-1]
                    self.list_rgb.append(sub_dir0 + "/rgb/{}.png".format(input_line))
                    self.list_depth.append(sub_dir0 + "/depth/{}.h5".format(input_line))
                    self.list_label.append(sub_dir0 + "/mask/{}.png".format(input_line))
                    self.list_obj.append(obj_category)
                    self.list_instance.append(ins)
                    self.list_status.append(art_index)
                    self.list_rank.append(int(input_line))
                try:
                    meta_file = open(sub_dir0 + "/gt.yml", "r")
                    meta_instance = yaml.load(meta_file)
                    meta[art_index] = meta_instance
                except:
                    meta[art_index] = None

            # Deal with different dataset in this part to parse the URDF
            urdf_path = self.root_dir + "/urdf/" + obj_category + "/" + ins
            if args.dataset == "sapien":
                urdf_ins = get_urdf_mobility(urdf_path, False)
            else:
                urdf_ins = get_urdf(urdf_path)
            # root_urdf     = tree_urdf.getroot()
            # rpy_xyz       = {}
            # list_xyz      = [None] * self.max_lnk
            # list_rpy      = [None] * self.max_lnk
            # list_box      = [None] * self.max_lnk
            # # ['obj'] ['link/joint']['xyz/rpy'] [0, 1, 2, 3, 4]
            # num_links     = 0
            # for link in root_urdf.iter('link'):
            #     num_links += 1
            #     index_link = None

            #     if link.attrib['name']=='base_link':
            #         index_link = 0
            #     else:
            #         index_link = int(link.attrib['name'])
            #     for visual in link.iter('visual'):
            #         for origin in visual.iter('origin'):
            #             list_xyz[index_link] = [float(x) for x in origin.attrib['xyz'].split()]
            #             list_rpy[index_link] = [float(x) for x in origin.attrib['rpy'].split()]

            # rpy_xyz['xyz']   = list_xyz
            # rpy_xyz['rpy']   = list_rpy
            # # rpy_xyz['box']   = list_box
            # urdf_ins['link'] = rpy_xyz

            # rpy_xyz       = {}
            # list_xyz      = [None] * self.max_lnk
            # list_rpy      = [None] * self.max_lnk
            # list_axis     = [None] * self.max_lnk
            # # here we still have to read the URDF file
            # for joint in root_urdf.iter('joint'):
            #     index_joint = int(joint.attrib['name'][0])
            #     for origin in joint.iter('origin'):
            #         list_xyz[index_joint] = [float(x) for x in origin.attrib['xyz'].split()]
            #         list_rpy[index_joint] = [float(x) for x in origin.attrib['rpy'].split()]
            #     for axis in joint.iter('axis'):
            #         list_axis[index_joint]= [float(x) for x in axis.attrib['xyz'].split()]
            # rpy_xyz['xyz']       = list_xyz
            # rpy_xyz['rpy']       = list_rpy
            # rpy_xyz['axis']      = list_axis

            # urdf_ins['joint']    = rpy_xyz
            # urdf_ins['num_links']= num_links

            meta_dict_obj[ins] = meta
            urdf_dict_obj[ins] = urdf_ins

            print("Object {} instance {} buffer loaded".format(obj_category, ins))

            self.meta_dict[obj_category] = meta_dict_obj
            self.urdf_dict[obj_category] = urdf_dict_obj

        self.length = len(self.list_rgb)
        self.height = 512
        self.width = 512
        self.xmap = np.array([[j for i in range(512)] for j in range(512)])
        self.ymap = np.array([[i for i in range(512)] for j in range(512)])

        self.num = num_points
        self.add_noise = add_noise

    def __len__(self):
        return self.length

    def __preprocess_and_save__(self, index):
        obj_category = self.list_obj[index]
        ins = self.list_instance[index]
        obj = self.objnamelist.index(obj_category)
        art_status = self.list_status[index]
        frame_order = self.list_rank[index]
        label = self.list_label[index]
        h5_save_path = (
            self.root_dir + "/hdf5/" + obj_category + "/" + ins + "/" + art_status
        )
        if not os.path.exists(h5_save_path):
            os.makedirs(h5_save_path)
        h5_save_name = h5_save_path + "/{}.h5".format(frame_order)
        num_parts = self.urdf_dict[obj_category][ins]["num_links"]
        new_num_parts = num_parts - 1

        model_offsets = self.urdf_dict[obj_category][ins]["link"]
        joint_offsets = self.urdf_dict[obj_category][ins]["joint"]

        parts_model_point = [None] * new_num_parts
        parts_world_point = [None] * new_num_parts
        parts_target_point = [None] * new_num_parts

        parts_cloud_cam = [None] * new_num_parts
        parts_cloud_world = [None] * new_num_parts
        parts_cloud_canon = [None] * new_num_parts
        parts_cloud_urdf = [None] * new_num_parts
        parts_cloud_norm = [None] * new_num_parts

        parts_world_pos = [None] * new_num_parts
        parts_world_orn = [None] * new_num_parts
        parts_urdf_pos = [None] * new_num_parts
        parts_urdf_orn = [None] * new_num_parts
        parts_urdf_box = [None] * new_num_parts

        parts_model2world = [None] * new_num_parts
        parts_canon2urdf = [None] * new_num_parts
        parts_target_r = [None] * new_num_parts
        parts_target_t = [None] * new_num_parts

        parts_mask = [None] * new_num_parts
        choose_x = [None] * new_num_parts
        choose_y = [None] * new_num_parts
        choose_to_whole = [None] * new_num_parts

        # rgb/depth/label
        print("current image: ", self.list_rgb[index])
        img = Image.open(self.list_rgb[index])
        img = np.array(img)  # .astype(np.uint8)
        depth = np.array(h5py.File(self.list_depth[index], "r")["data"])
        label = np.array(Image.open(self.list_label[index]))

        # pose infos
        pose_dict = self.meta_dict[obj_category][ins][art_status][
            "frame_{}".format(frame_order)
        ]
        urdf_dict = self.urdf_dict[obj_category][ins]
        viewMat = np.array(pose_dict["viewMat"]).reshape(4, 4).T
        projMat = np.array(pose_dict["projMat"]).reshape(4, 4).T

        parts_world_pos[0] = np.array([0, 0, 0])
        parts_world_orn[0] = np.array([0, 0, 0, 1])
        # SAPIEN: skip the virtual base part
        for link in range(1, num_parts):
            if link > 0:
                parts_world_pos[link - 1] = np.array(
                    pose_dict["obj"][link - 1][4]
                ).astype(np.float32)
                parts_world_orn[link - 1] = np.array(
                    pose_dict["obj"][link - 1][5]
                ).astype(np.float32)

        for link in range(1, num_parts):
            if link == 1 and num_parts == 2:
                parts_urdf_pos[link] = np.array(
                    urdf_dict["joint"]["xyz"][link - 1]
                )  # todo, accumulate joints offsets != link offsets
            else:
                # Only change this branch for SAPIEN data
                parts_urdf_pos[link - 1] = -np.array(urdf_dict["link"]["xyz"][link][0])
            parts_urdf_orn[link - 1] = np.array(urdf_dict["link"]["rpy"][link][0])

        for k in range(1, num_parts):
            center_world_orn = parts_world_orn[k - 1]
            center_world_orn = np.array(
                [
                    center_world_orn[3],
                    center_world_orn[0],
                    center_world_orn[1],
                    center_world_orn[2],
                ]
            )
            my_model2world_r = quaternion_matrix(center_world_orn)[
                :4, :4
            ]  # [w, x, y, z]
            my_model2world_t = parts_world_pos[k - 1]
            my_model2world_mat = np.copy(my_model2world_r)
            for m in range(3):
                my_model2world_mat[m, 3] = my_model2world_t[m]
            my_world2camera_mat = viewMat
            my_camera2clip_mat = projMat
            my_model2camera_mat = np.dot(my_world2camera_mat, my_model2world_mat)
            parts_model2world[k - 1] = my_model2world_mat

        # depth to cloud data
        # SAPIEN, throw away the label 0 for virtual base link
        mask = np.array((label[:, :] < num_parts) & (label[:, :] > 0)).astype(np.uint8)
        mask_whole = np.copy(mask)
        for n in range(1, num_parts):
            parts_mask[n - 1] = np.array((label[:, :] == (n))).astype(np.uint8)
            choose_to_whole[n - 1] = np.where(parts_mask[n - 1] > 0)

        # >>>>>>>>>>------- rendering target pcloud from depth image --------<<<<<<<<<#
        # first get projected map
        # ymap = self.ymap
        # xmap = self.xmap
        # h = self.height
        # w = self.width
        # u_map = ymap * 2 / w - 1
        # v_map = (512 - xmap) * 2 / h - 1
        # v1_map = xmap * 2 / h - 1
        # w_channel = -depth
        # projected_map = np.stack(
        #     [u_map * w_channel, v_map * w_channel, depth, w_channel]
        # ).transpose([1, 2, 0])
        # projected_map1 = np.stack(
        #     [u_map * w_channel, v1_map * w_channel, depth, w_channel]
        # ).transpose([1, 2, 0])
        for s in range(1, num_parts):
            x_set, y_set = choose_to_whole[s - 1]
            if len(x_set) < 10:
                print(ins, art_status, frame_order)
                print(
                    "data is empty, skipping!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )
                return None
            else:
                choose_x[s - 1] = x_set
                choose_y[s - 1] = y_set

            # ---------------> from projected map into target part_cloud
            # order: cam->world->canon)

            # MotionNet Modification
            point_num = np.shape(choose_x[s - 1])[0]
            cloud_cam = []
            cloud_cam_real = []
            for i in range(point_num):
                x = choose_x[s - 1][i]
                y = choose_y[s - 1][i]
                y_real = 512 - y
                cloud_cam.append(
                    [
                        (y + projMat[0, 2]) * depth[x, y] / projMat[0, 0],
                        -(x + projMat[1, 2]) * depth[x, y] / (-projMat[1, 1]),
                        -depth[x, y],
                    ]
                )
                cloud_cam_real.append(
                    [
                        (y + projMat[0, 2]) * depth[x, y] / projMat[0, 0],
                        -(x + projMat[1, 2]) * depth[x, y] / (-projMat[1, 1]),
                        -depth[x, y],
                    ]
                )
                # cloud_cam.append(
                #     [
                #         (x + projMat[0, 2]) * depth[x, y_real] / projMat[0, 0],
                #         -(y_real + projMat[1, 2]) * depth[x, y_real] / (-projMat[1, 1]),
                #         -depth[x, y_real],
                #     ]
                # )
                # pdb.set_trace()
                # print(cloud_cam)
                # break
            cloud_cam = np.array(cloud_cam)
            cloud_cam_real = np. array(cloud_cam_real)

            # pdb.set_trace()
            # MotionNet Modification
            # projected_points = projected_map[choose_x[s-1][:].astype(np.uint16), choose_y[s-1][:].astype(np.uint16), :]
            # projected_points = np.reshape(projected_points, [-1, 4])
            # depth_channel    = - projected_points[:, 3:4]
            # cloud_cam        = np.dot(projected_points[:, 0:2] - np.dot(depth_channel, projMat[0:2, 2:3].T), np.linalg.pinv(projMat[:2, :2].T))

            # projected_points1 = projected_map1[choose_x[s-1][:].astype(np.uint16), choose_y[s-1][:].astype(np.uint16), :]
            # projected_points1 = np.reshape(projected_points1, [-1, 4])
            # cloud_cam_real    = np.dot(projected_points1[:, 0:2] - np.dot(depth_channel, projMat[0:2, 2:3].T), np.linalg.pinv(projMat[:2, :2].T))
            # cloud_cam_real    = np.concatenate((cloud_cam_real, depth_channel), axis=1)

            # cloud_cam      = np.concatenate((cloud_cam, depth_channel), axis=1)



            cloud_cam_full = np.concatenate(
                (cloud_cam, np.ones((cloud_cam.shape[0], 1))), axis=1
            )

            # modify, todo
            camera_pose_mat = np.linalg.pinv(viewMat.T)
            # MotionNet modification
            # camera_pose_mat[:3, :] = -camera_pose_mat[:3, :]
            cloud_world = np.dot(cloud_cam_full, camera_pose_mat)
            cloud_canon = np.dot(
                cloud_world, np.linalg.pinv(parts_model2world[s - 1].T)
            )

            # canon points should be points coordinates centered in the inertial frame
            parts_cloud_cam[s - 1] = cloud_cam_real[:, :3]
            parts_cloud_world[s - 1] = cloud_world[:, :3]
            parts_cloud_canon[s - 1] = cloud_canon[:, :3]

        for k in range(1, num_parts):
            center_joint_orn = parts_urdf_orn[k - 1]
            my_canon2urdf_r = euler_matrix(
                center_joint_orn[0], center_joint_orn[1], center_joint_orn[2]
            )[
                :4, :4
            ]  # [w, x, y, z]
            my_canon2urdf_t = parts_urdf_pos[k - 1]
            my_canon2urdf_mat = my_canon2urdf_r
            for m in range(3):
                my_canon2urdf_mat[m, 3] = my_canon2urdf_t[m]
            part_points_space = np.concatenate(
                (
                    parts_cloud_canon[k - 1],
                    np.ones((parts_cloud_canon[k - 1].shape[0], 1)),
                ),
                axis=1,
            )
            parts_cloud_urdf[k - 1] = np.dot(part_points_space, my_canon2urdf_mat.T)

        # >>>>>>>>>>>>>>> go to NPCS space
        # NPCS here is not stored into hdf5
        for link in range(1, num_parts):
            tight_w = max(parts_cloud_urdf[link - 1][:, 0]) - min(
                parts_cloud_urdf[link - 1][:, 0]
            )
            tight_l = max(parts_cloud_urdf[link - 1][:, 1]) - min(
                parts_cloud_urdf[link - 1][:, 1]
            )
            tight_h = max(parts_cloud_urdf[link - 1][:, 2]) - min(
                parts_cloud_urdf[link - 1][:, 2]
            )
            norm_factor = np.sqrt(1) / np.sqrt(
                tight_w ** 2 + tight_l ** 2 + tight_h ** 2
            )
            base_p = np.array(
                [
                    min(parts_cloud_urdf[link - 1][:, 0]),
                    min(parts_cloud_urdf[link - 1][:, 1]),
                    min(parts_cloud_urdf[link - 1][:, 2]),
                ]
            ).reshape(1, 3)
            extre_p = np.array(
                [
                    max(parts_cloud_urdf[link - 1][:, 0]),
                    max(parts_cloud_urdf[link - 1][:, 1]),
                    max(parts_cloud_urdf[link - 1][:, 2]),
                ]
            ).reshape(1, 3)
            center_p = (extre_p - base_p) / 2 * norm_factor

            parts_cloud_norm[link - 1] = (
                (parts_cloud_urdf[link - 1][:, :3] - base_p) * norm_factor
                + np.array([0.5, 0.5, 0.5]).reshape(1, 3)
                - center_p.reshape(1, 3)
            )

        x_set_pcloud = np.concatenate(choose_x, axis=0)
        y_set_pcloud = np.concatenate(choose_y, axis=0)

        # save into h5 for rgb_img, input_pts, mask, correpsonding urdf_points
        print("Writing to ", h5_save_name)
        hf = h5py.File(h5_save_name, "w")
        hf.create_dataset("rgb", data=img)
        hf.create_dataset("mask", data=mask)
        cloud_cam = hf.create_group("gt_points")
        for part_i, points in enumerate(parts_cloud_cam):
            cloud_cam.create_dataset(str(part_i), data=points)
        coord_gt = hf.create_group("gt_coords")
        for part_i, points in enumerate(parts_cloud_urdf):
            coord_gt.create_dataset(str(part_i), data=points)
        hf.close()

        ################# for debug only, let me know if you have questions #################
        if self.is_debug:
            figure = plt.figure(dpi=200)
            ax = plt.subplot(121)
            plt.imshow(img)
            plt.title('RGB image')
            ax1 = plt.subplot(122)
            plt.imshow(depth)
            plt.title('depth image')
            plt.show()
            plot3d_pts([parts_cloud_cam], [['part {}'.format(i) for i in range(len(parts_cloud_cam))]], s=5, title_name=['camera coords'])
            plot3d_pts([parts_cloud_world], [['part {}'.format(i) for i in range(len(parts_cloud_world))]], s=5, title_name=['world coords'])
            plot3d_pts([parts_cloud_canon], [['part {}'.format(i) for i in range(len(parts_cloud_canon))]], s=5, title_name=['canon coords'])
            plot3d_pts([parts_cloud_urdf], [['part {}'.format(i) for i in range(len(parts_cloud_urdf))]], s=5, title_name=['urdf coords'])

        return None

def run(PoseData, index):
    PoseData.__preprocess_and_save__(index)

if __name__ == "__main__":
    # >>>>>>>>>>>>>>>>>>>>>>>>> config here >>>>>>>>>>>>>>>>>>>>>>>#
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="shape2motion", help="name of the dataset we use"
    )  # todo
    parser.add_argument(
        "--item", default="eyeglasses", help="name of the dataset we use"
    )
    parser.add_argument(
        "--num_expr", default="0.01", help="get configuration file per expriment"
    )
    parser.add_argument(
        "--mode", default="train", help="indicating whether in demo mode"
    )
    parser.add_argument(
        "--debug", action="store_true", help="indicating whether in debug mode"
    )
    args = parser.parse_args()

    name_dataset = args.dataset
    item = args.item
    infos = global_info()
    my_dir = infos.base_path
    root_dset = my_dir + "/dataset/" + name_dataset
    # selected_list = infos.datasets[item].train_list # default None, if specifies, will only choose specified instances
    selected_list = None
    # >>>>>>>>>>>>>>>>>>>>>>>>> config end here >>>>>>>>>>>>>>>>>>>#

    # 1. collect filenames into all.txt
    collect_file(root_dset, [item], mode=args.mode)
    PoseData = PoseDataset(
        root_dset,
        item,
        is_debug=args.debug,
        mode=args.mode,
        selected_list=selected_list,
    )
    print("number of images: ", len(PoseData.list_rgb))

    # # 2. preprocess and save
    pool = multiprocessing.Pool(processes=16)
    for i in range(0, len(PoseData.list_rgb)):
        # print(PoseData.list_rgb[i])
        # data = PoseData.__preprocess_and_save__(i)
        pool.apply_async(run, (PoseData, i,))
    

    pool.close()
    pool.join()
    # 3. split data into train & test
    split_dataset(root_dset, [item], args, test_ins=infos.datasets[item].test_list, spec_ins=infos.datasets[item].spec_list, train_ins=infos.datasets[item].train_list)

    stop = time.time()
    print(str(stop - start) + " seconds")
