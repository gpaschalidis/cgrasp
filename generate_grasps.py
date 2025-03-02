# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Universiteit van Amsterdam (UvA).
# All rights reserved.
#
# Universiteit van Amsterdam (UvA) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with UvA or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: g.paschalidis@uva.nl
#

import sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
import open3d as o3d
import torch
import os, time
import argparse
import random
import mano
from tqdm import tqdm
from cgrasp.tools.utils import euler
from cgrasp.tools.cfg_parser import Config
from cgrasp.tests.tester import Tester

from cgrasp.tools.train_tools import point2point_signed
from cgrasp.tools.utils import aa2rotmat
from cgrasp.tools.utils import to_cpu
from cgrasp.tools.meshviewer import points2sphere, Mesh

from bps_torch.bps import bps_torch
from vis_utils import create_o3d_mesh, create_line_set

def generate_random_vector():
    """
    Generates a random 3D vector.
    """
    vector = np.random.randn(3)
    while np.linalg.norm(vector) == 0:
        vector = np.random.randn(3)
    return vector / np.linalg.norm(vector)

def vis_meshes(dorig, 
               cgrasp, 
               refine_net, 
               rh_model, 
               n_samples=10, 
               obj_name=None):

    with torch.no_grad():

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        dir_ = [generate_random_vector() for i in range(n_samples)]
        dir_ = np.array(dir_)
        
        dir_ = torch.tensor(dir_).to(torch.float32).to(device)
        
        cgrasp_out = cgrasp.sample_poses(dorig['bps_object'], dir_hand=dir_)

        output = rh_model(**cgrasp_out)
        verts_rh_gen_cgrasp = output.vertices

        _, h2o, _ = point2point_signed(verts_rh_gen_cgrasp, dorig['verts_object'].to(device))
        cgrasp_out['trans_rhand_f'] = cgrasp_out['transl']
        cgrasp_out['global_orient_rhand_rotmat_f'] = aa2rotmat(cgrasp_out['global_orient']).view(-1, 3, 3)
        cgrasp_out['fpose_rhand_rotmat_f'] = aa2rotmat(cgrasp_out['hand_pose']).view(-1, 15, 3, 3)
        cgrasp_out['verts_object'] = dorig['verts_object'].to(device)
        cgrasp_out['h2o_dist'] = h2o.abs()

        drec_rnet = refine_net(**cgrasp_out)
        verts_rh_gen_rnet = rh_model(**drec_rnet).vertices
        
        renderables = []
        gen_meshes = []
        for cId in range(0, len(dorig['bps_object'])):
            obj_mesh = dorig['mesh_object'][cId]
            hand_mesh_gen_rnet = Mesh(vertices=to_cpu(verts_rh_gen_rnet[cId]), faces=rh_model.faces, vc=[245, 191, 177])

            if 'rotmat' in dorig:
                rotmat = dorig['rotmat'][cId].T
                obj_mesh = obj_mesh.rotate_vertices(rotmat)
                hand_mesh_gen_rnet.rotate_vertices(rotmat)

            gen_meshes.append([obj_mesh, hand_mesh_gen_rnet])
            obj_verts = np.array(obj_mesh.vertices)
            obj_faces = np.array(obj_mesh.faces)
            obj_dims = obj_verts.max(0) - obj_verts.min(0)
            rhand_verts = np.array(hand_mesh_gen_rnet.vertices)
            rhand_faces = np.array(hand_mesh_gen_rnet.faces)
            new_obj_mesh = create_o3d_mesh(obj_verts, obj_faces, [0.9, 0.4, 0.3])
            new_rhand_mesh = create_o3d_mesh(rhand_verts, rhand_faces, [0.8, 0.1, 0.3])
            origin  = (obj_verts.max(0) + obj_verts.min(0)) / 2
            target_dir = (dir_[cId].cpu().numpy() @ rotmat.T)
            line = create_line_set(origin[None], origin[None] + 0.5 * target_dir, [0,0,1])
            o3d.visualization.draw_geometries([new_obj_mesh, new_rhand_mesh, line])
             

def grab_new_objs(cgrasp_tester, 
                  obj_path, 
                  save_dir,
                  rot=False,
                  n_samples=10, 
                  scale=1.): 

    cgrasp_tester.cgrasp.eval()
    cgrasp_tester.refine_net.eval()
    device = cgrasp_tester.device
    rh_model = mano.load(model_path=cgrasp_tester.cfg.rhm_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=n_samples,
                         flat_hand_mean=True).to(device)

    cgrasp_tester.refine_net.rhm_train = rh_model

    bps = bps_torch(custom_basis=cgrasp_tester.bps)

    obj_name = obj_path.split("/")[-1].split(".")[0]
    obj = obj_path.split('/')[-1][:-4]
    if rot:
        rand_rotdeg = np.random.random([n_samples, 3]) * np.array([360, 360, 360])
    else:
        rand_rotdeg = np.random.random([n_samples, 3]) * np.array([0, 0, 0])

    rand_rotmat = euler(rand_rotdeg)
    
    dorig = {'bps_object': [], 
             'verts_object': [], 
             'mesh_object': [], 
             'rotmat': []} 

    
    for samples in range(n_samples):
        verts_obj, mesh_obj, rotmat = load_obj_verts(obj_path, rand_rotmat[samples], rndrotate=rot, scale=scale)
        bps_object = bps.encode(torch.from_numpy(verts_obj), feature_type='dists')['dists']
        dorig['bps_object'].append(bps_object.to(device))

        dorig['verts_object'].append(torch.from_numpy(verts_obj.astype(np.float32)).unsqueeze(0))
        dorig['mesh_object'].append(mesh_obj)
        dorig['rotmat'].append(rotmat)
   
    dorig['bps_object'] = torch.cat(dorig['bps_object'])
    dorig['verts_object'] = torch.cat(dorig['verts_object'])
    vis_meshes(dorig=dorig,
               cgrasp=cgrasp_tester.cgrasp.to(device),
               refine_net=cgrasp_tester.refine_net.to(device),
               rh_model=rh_model,
               n_samples=n_samples,
               obj_name=obj_name)

def load_obj_verts(mesh_path, rand_rotmat, rndrotate=True, scale=1., n_sample_verts=2048):
    np.random.seed(100)
    obj_mesh = Mesh(filename=mesh_path, vscale=scale)


    ## center and scale the object
    max_length = np.linalg.norm(obj_mesh.vertices, axis=1).max()
    if max_length > .3:
        re_scale = max_length / .08
        print(f'The object is very large, down-scaling by {re_scale} factor')
        obj_mesh.vertices[:] = obj_mesh.vertices / re_scale

    object_fullpts = obj_mesh.vertices
    maximum = object_fullpts.max(0, keepdims=True)
    minimum = object_fullpts.min(0, keepdims=True)
    offset = (maximum + minimum) / 2
    verts_obj = object_fullpts - offset
    obj_mesh.vertices[:] = verts_obj

    if rndrotate:
        obj_mesh.rotate_vertices(rand_rotmat)
    else:
        rand_rotmat = np.eye(3)

    while (obj_mesh.vertices.shape[0]<n_sample_verts):
        new_mesh = obj_mesh.subdivide()
        obj_mesh = Mesh(vertices=new_mesh.vertices,
                        faces = new_mesh.faces,
                        visual = new_mesh.visual)

    verts_obj = obj_mesh.vertices
    verts_sample_id = np.random.choice(verts_obj.shape[0], n_sample_verts, replace=False)
    verts_sampled = verts_obj[verts_sample_id]

    return verts_sampled, obj_mesh, rand_rotmat

def main():
    parser = argparse.ArgumentParser(description='GrabNet-Testing')

    parser.add_argument('obj_path',   
                        type=str,
                        help='The path to the 3D object Mesh or Pointcloud dataset'
    )
    parser.add_argument('rhm_path', 
                        type=str,
                        help='The path to the folder containing MANO_RIHGT model'
    )
    parser.add_argument('save_dir', 
                        type=str,
                        help='The path where the results will be stored'
    )
    parser.add_argument('--scale', 
                        default=1., 
                        type=float,
                        help='The scaling for the 3D object'
    )
    parser.add_argument('--n_samples', 
                        default=10, 
                        type=int,
                        help='number of grasps to generate'
    )
    parser.add_argument('--config_path', 
                        default='cgrasp/pretrained/cgrasp_cfg.yaml', 
                        type=str,
                        help='The path to the confguration of the pre trained CGrasp model'
    )
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"    

    save_dir = args.save_dir
    obj_path = args.obj_path
    rhm_path = args.rhm_path
    scale = args.scale
    n_samples = args.n_samples
    cfg_path = args.config_path
    
    best_rnet = 'cgrasp/pretrained/refinenet.pt'
    bps_dir = 'cgrasp/configs/bps.npz'

    config = {
        'save_dir': save_dir,
        'bps_dir': bps_dir,
        'rhm_path': rhm_path,
        'best_rnet': best_rnet
    }
    
    cfg = Config(default_cfg_path=cfg_path, **config)

    cgrasp_tester = Tester(cfg=cfg)
    grab_new_objs(cgrasp_tester, 
                  obj_path,
                  save_dir,
                  rot=True, 
                  n_samples=n_samples, 
                  scale=scale) 


if __name__ == '__main__':
    main()
