import json
import sys
import os
import time
import numpy as np
import skimage.transform
import networkx as nx
import cv2

import matplotlib.pyplot as plt

working_directory = os.getcwd()

trajectory_path = working_directory + '/tasks/R2R-pano/results/non-blind/regretful-agent-data|synthetic_resume|best-real_val_unseen_epoch_335.json'

instruction_path = working_directory + '/tasks/R2R-pano/data/R2R_val_unseen.json'

graph_path = working_directory + '/connectivity/%s_connectivity.json'

# 80 is the typical
idx = 29  # the trajectory index to visualize

# load navigation graph to calculate the relative heading of the next location
def load_nav_graph(graph_path):
    with open(graph_path, "r") as f:
        G = nx.Graph()
        positions = {}
        data = json.load(f)
        for i,item in enumerate(data):
            if item['included']:
                for j,conn in enumerate(item['unobstructed']):
                    if conn and data[j]['included']:
                        positions[item['image_id']] = np.array([item['pose'][3], 
                                item['pose'][7], item['pose'][11]]);
                        assert data[j]['unobstructed'][i], 'Graph should be undirected'
                        G.add_edge(item['image_id'],data[j]['image_id'])
        nx.set_node_attributes(G, values=positions, name='position')
    return G

def compute_rel_heading(graph, current_viewpoint, current_heading, next_viewpoint):
    if current_viewpoint == next_viewpoint:
        return 0.
    target_rel = graph.nodes[next_viewpoint]['position'] - graph.nodes[current_viewpoint]['position']
    target_heading = np.pi/2.0 - np.arctan2(target_rel[1], target_rel[0]) # convert to rel to y axis
    
    rel_heading = target_heading - current_heading
    # normalize angle into turn into [-pi, pi]
    rel_heading = rel_heading - (2*np.pi) * np.floor((rel_heading + np.pi) / (2*np.pi))
    return rel_heading


with open(trajectory_path, "r") as f:
    trajectory_data = json.load(f)
with open(instruction_path, "r") as f:
    instruction_data = json.load(f)
    
instr_id2txt = {
    ('%s_%d' % (d['path_id'], n)): txt for d in instruction_data for n, txt in enumerate(d['instructions'])}
instr_id2scan = {
    ('%s_%d' % (d['path_id'], n)): d['scan'] for d in instruction_data for n, txt in enumerate(d['instructions'])}

graphs = {scan: load_nav_graph(graph_path % scan) for scan in instr_id2scan.values()}


def visualize_panorama_img(scan, viewpoint, heading, elevation):
    WIDTH = 80
    HEIGHT = 480
    pano_img = np.zeros((HEIGHT, WIDTH*36, 3), np.uint8)
    VFOV = np.radians(55)
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.init()
    for n_angle, angle in enumerate(range(-175, 180, 10)):
        sim.newEpisode(scan, viewpoint, heading + np.radians(angle), elevation)
        state = sim.getState()
        im = state.rgb
        pano_img[:, WIDTH*n_angle:WIDTH*(n_angle+1), :] = im[..., ::-1]
    return pano_img

def visualize_tunnel_img(scan, viewpoint, heading, elevation):
    
    WIDTH = 640
    HEIGHT = 480
    VFOV = np.radians(60)
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.init()
    sim.newEpisode(scan, viewpoint, heading, elevation)
    state = sim.getState()
    im = state.rgb
    return im[..., ::-1].copy()

def load_image(instr, viewpoint):
    HEIGHT = 1024
    WIDTH = 1024
    pano_img = np.zeros((HEIGHT, WIDTH*4, 3), np.uint8)

    for i in range(4):
        image_path = viewpoint + f"_skybox{i + 1}_sami.jpg"
        abs_path = "/mnt/10802E8A802E75FE/Magister/topicos_ia/Proyecto/Matterport3D/data/v1/scans"
        rel_path = f"{abs_path}/{instr}/matterport_skybox_images/{image_path}"
        im = cv2.imread(rel_path)  
        pano_img[:, WIDTH*i:WIDTH*(i+1), :] = im[..., ::-1]
    return pano_img

## Code

trajectory = trajectory_data[idx]
# print(trajectory.keys())
instr_id = trajectory['instr_id']
# print(trajectory['trajectory'], len(trajectory['trajectory']))
print(trajectory['img_attn'], len(trajectory['img_attn'][0]))
scan = instr_id2scan[instr_id]
txt = instr_id2txt[instr_id]

graph = graphs[scan]

plt.close('all')
print(txt)


for n, (viewpoint, heading, elevation) in enumerate(trajectory['trajectory']):

    im = load_image(scan, viewpoint)
    plt.figure(figsize=(18, 3))
    plt.imshow(im)
    plt.xticks(np.linspace(0, im.shape[1] - 1, 5), [-180, -90, 0, 90, 180])
    plt.xlabel('relative heading from the agent')
    plt.yticks([], [])
    plt.title('step %d panorama view' % n)
    if n + 1 < len(trajectory['trajectory']):
        next_viewpoint, _, _ = trajectory['trajectory'][n+1]
        if next_viewpoint != viewpoint:
            rel_heading = compute_rel_heading(graph, viewpoint, heading, next_viewpoint)
            next_im_x = (rel_heading / (2*np.pi) + 0.5) * im.shape[1]
            # Not working
            plt.arrow(next_im_x, im.shape[0] - 10, 0, -50, width=10, color='r')
        else:
            plt.text(im.shape[1] // 2 - 60, im.shape[0] - 30, 'Stop', fontsize=20, color='r')
    plt.show()

