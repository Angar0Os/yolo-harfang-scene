import harfang as hg
import math
import cv2
import ctypes
import numpy as np
import matplotlib.pyplot as plt 
import mediapipe as mp
import time

def create_transformation_matrix(translation, scale, rotation):
    T = np.eye(4)
    T[:3, 3] = translation
    
    rx, ry, rz = rotation
    Rx = np.array([[1, 0, 0, 0], [0, np.cos(rx), -np.sin(rx), 0], [0, np.sin(rx), np.cos(rx), 0], [0, 0, 0, 1]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry), 0], [0, 1, 0, 0], [-np.sin(ry), 0, np.cos(ry), 0], [0, 0, 0, 1]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0, 0], [np.sin(rz), np.cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    S = np.eye(4)
    S[0, 0], S[1, 1], S[2, 2] = scale[0], scale[1], scale[2]
    
    return T @ Rz @ Ry @ Rx @ S

def apply_bone_correction(position, correction):
    if position is None or correction is None:
        return position
    
    transform_matrix = create_transformation_matrix(
        correction['translation'], 
        correction['scale'], 
        correction['rotation']
    )
    pos_homogeneous = np.array([position[0], position[1], position[2], 1.0])
    corrected_pos = transform_matrix @ pos_homogeneous
    return corrected_pos[:3].tolist()

def calculate_optimal_correction(real_positions, triangulated_positions):
    """Calcule la correction optimale pour une superposition parfaite"""
    if len(real_positions) < 10 or len(triangulated_positions) < 10:
        return None
    
    real_array = np.array([[pos.x, pos.y, pos.z] for pos in real_positions])
    tri_array = np.array(triangulated_positions)
    
    min_len = min(len(real_array), len(tri_array))
    real_array = real_array[:min_len]
    tri_array = tri_array[:min_len]
    
    real_center = np.mean(real_array, axis=0)
    tri_center = np.mean(tri_array, axis=0)
    translation_offset = real_center - tri_center
    
    real_centered = real_array - real_center
    tri_centered = tri_array - tri_center
    
    scale_factor = np.ones(3)
    for i in range(3):
        real_std = np.std(real_centered[:, i])
        tri_std = np.std(tri_centered[:, i])
        if tri_std > 1e-6:  
            scale_factor[i] = real_std / tri_std
    
    for i in range(3):
        if np.std(real_centered[:, i]) > 1e-6 and np.std(tri_centered[:, i]) > 1e-6:
            correlation = np.corrcoef(real_centered[:, i], tri_centered[:, i])[0, 1]
            if correlation < 0:
                scale_factor[i] *= -1
    
    return {
        'translation': translation_offset,
        'scale': scale_factor,
        'rotation': np.array([0.0, 0.0, 0.0])
    }

def apply_corrections_to_all_positions(triangulated_positions, correction):
    if correction is None:
        return triangulated_positions
    
    corrected_positions = []
    for pos in triangulated_positions:
        corrected_pos = apply_bone_correction(pos, correction)
        corrected_positions.append(corrected_pos)
    
    return corrected_positions

def calculate_global_bounds(all_bone_data):
    """Calcule les limites globales pour tous les os afin d'avoir la même échelle"""
    global_min = np.array([float('inf'), float('inf'), float('inf')])
    global_max = np.array([float('-inf'), float('-inf'), float('-inf')])
    
    for bone_key, data in all_bone_data.items():
        if not data['real_positions'] or not data['triangulated_positions']:
            continue
            
        real_array = np.array([[pos.x, pos.y, pos.z] for pos in data['real_positions']])
        global_min = np.minimum(global_min, np.min(real_array, axis=0))
        global_max = np.maximum(global_max, np.max(real_array, axis=0))
        
        if data['triangulated_positions']:
            tri_array = np.array(data['triangulated_positions'])
            global_min = np.minimum(global_min, np.min(tri_array, axis=0))
            global_max = np.maximum(global_max, np.max(tri_array, axis=0))
        
        if data['corrected_positions']:
            corr_array = np.array(data['corrected_positions'])
            global_min = np.minimum(global_min, np.min(corr_array, axis=0))
            global_max = np.maximum(global_max, np.max(corr_array, axis=0))
    
    margin = (global_max - global_min) * 0.1
    global_min -= margin
    global_max += margin
    
    return global_min, global_max

def show_comparison_plot_with_same_scale(real_positions, triangulated_positions, corrected_positions, bone_name, global_bounds):
    """Affiche les graphiques de comparaison avec la même échelle pour tous"""
    if not real_positions or not triangulated_positions:
        return
        
    global_min, global_max = global_bounds
    
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(f'Analyse Complète - {bone_name} (Échelle Unifiée)', fontsize=16, fontweight='bold')

    ax1 = fig.add_subplot(141, projection='3d')
    real_x = [pos.x for pos in real_positions]
    real_y = [pos.y for pos in real_positions]
    real_z = [pos.z for pos in real_positions]
    ax1.plot(real_x, real_y, real_z, label="Réelle", color='green', linewidth=3, marker='o', markersize=3)
    ax1.set_title("Trajectoire Réelle")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(global_min[0], global_max[0])
    ax1.set_ylim(global_min[1], global_max[1])
    ax1.set_zlim(global_min[2], global_max[2])
    ax1.legend()

    ax2 = fig.add_subplot(142, projection='3d')
    tri_x = [pos[0] for pos in triangulated_positions]
    tri_y = [pos[1] for pos in triangulated_positions]
    tri_z = [pos[2] for pos in triangulated_positions]
    ax2.plot(tri_x, tri_y, tri_z, label="Triangulée", color='red', linewidth=2, marker='s', markersize=2)
    ax2.set_title("Triangulation Brute")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(global_min[0], global_max[0])
    ax2.set_ylim(global_min[1], global_max[1])
    ax2.set_zlim(global_min[2], global_max[2])
    ax2.legend()

    ax3 = fig.add_subplot(143, projection='3d')
    if corrected_positions:
        corr_x = [pos[0] for pos in corrected_positions]
        corr_y = [pos[1] for pos in corrected_positions]
        corr_z = [pos[2] for pos in corrected_positions]
        ax3.plot(corr_x, corr_y, corr_z, label="Corrigée", color='orange', linewidth=3, marker='^', markersize=3)
    ax3.set_title("Après Correction")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim(global_min[0], global_max[0])
    ax3.set_ylim(global_min[1], global_max[1])
    ax3.set_zlim(global_min[2], global_max[2])
    ax3.legend()
    
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.plot(real_x, real_y, real_z, label="Réelle", color='green', linewidth=4, alpha=0.8)
    ax4.plot(tri_x, tri_y, tri_z, label="Triangulée", color='red', linewidth=2, alpha=0.6, linestyle=':')
    if corrected_positions:
        ax4.plot(corr_x, corr_y, corr_z, label="Corrigée", color='orange', linewidth=2, alpha=0.9, linestyle='--')
    ax4.set_title("Superposition (Objectif: Courbes Parfaitement Alignées)")
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_xlim(global_min[0], global_max[0])
    ax4.set_ylim(global_min[1], global_max[1])
    ax4.set_zlim(global_min[2], global_max[2])
    ax4.legend()

    plt.tight_layout()
    plt.show()

def InitRenderToTexture(res, pipeline_texture_name="tex_rb", texture_name="tex_color_ref", res_x=800, res_y=800):
    frame_buffer = hg.CreateFrameBuffer(res_x, res_y, hg.TF_RGBA8, hg.TF_D24, 4, 'framebuffer')
    color = hg.GetColorTexture(frame_buffer)
    tex_color_ref = res.AddTexture(pipeline_texture_name, color)
    tex_readback = hg.CreateTexture(res_x, res_y, texture_name, hg.TF_ReadBack | hg.TF_BlitDestination, hg.TF_RGBA8)
    picture = hg.Picture(res_x, res_y, hg.PF_RGBA32)
    return frame_buffer, color, tex_color_ref, tex_readback, picture

def GetOpenCvImageFromPicture(picture):
    picture_width, picture_height = picture.GetWidth(), picture.GetHeight()
    picture_data = picture.GetData()
    bytes_per_pixels = 4
    data_size = picture_width * picture_height * bytes_per_pixels
    buffer = (ctypes.c_char * data_size).from_address(picture_data)
    raw_data = bytes(buffer)
    np_array = np.frombuffer(raw_data, dtype=np.uint8)
    image_rgba = np_array.reshape((picture_height, picture_width, bytes_per_pixels))
    return cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGR)

def get_landmark_position(landmarks, landmark_idx):
    if landmarks and landmark_idx < len(landmarks.landmark):
        return landmarks.landmark[landmark_idx]
    return None

def triangulate_position(point_front, point_side):
    if point_front is None or point_side is None:
        return None
    return [point_front.x - 0.5, point_front.y - 0.5, point_side.x - 0.5]

def draw_skeleton_on_image(image, landmarks, connections):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    if landmarks:
        mp_drawing.draw_landmarks(
            image,
            landmarks,
            connections,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    return image

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, 
                   min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_2 = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, 
                     min_detection_confidence=0.5, min_tracking_confidence=0.5)

bone_data = {
    'head': {'real_positions': [], 'triangulated_positions': [], 'corrected_positions': [], 
             'last_front': None, 'last_side': None, 'mediapipe_landmark': mp_pose.PoseLandmark.NOSE, 'name': 'Tête'},
    'right_arm': {'real_positions': [], 'triangulated_positions': [], 'corrected_positions': [], 
                  'last_front': None, 'last_side': None, 'mediapipe_landmark': mp_pose.PoseLandmark.RIGHT_SHOULDER, 'name': 'Bras Droit'},
    'left_arm': {'real_positions': [], 'triangulated_positions': [], 'corrected_positions': [], 
                 'last_front': None, 'last_side': None, 'mediapipe_landmark': mp_pose.PoseLandmark.LEFT_SHOULDER, 'name': 'Bras Gauche'},
    'right_leg': {'real_positions': [], 'triangulated_positions': [], 'corrected_positions': [], 
                  'last_front': None, 'last_side': None, 'mediapipe_landmark': mp_pose.PoseLandmark.RIGHT_HIP, 'name': 'Jambe Droite'},
    'left_leg': {'real_positions': [], 'triangulated_positions': [], 'corrected_positions': [], 
                 'last_front': None, 'last_side': None, 'mediapipe_landmark': mp_pose.PoseLandmark.LEFT_HIP, 'name': 'Jambe Gauche'},
    'hips': {'real_positions': [], 'triangulated_positions': [], 'corrected_positions': [], 
             'last_front': None, 'last_side': None, 'mediapipe_landmark': mp_pose.PoseLandmark.RIGHT_HIP, 'name': 'Hanches'}
}

hg.InputInit()
hg.WindowSystemInit()

res_x, res_y = 1920, 1080
win = hg.RenderInit('Détection de Pose avec MediaPipe en Temps Réel', res_x, res_y, hg.RF_VSync | hg.RF_MSAA8X)
hg.AddAssetsFolder("assets_compiled")

pipeline = hg.CreateForwardPipeline()
res = hg.PipelineResources()
frame = 0

scene = hg.Scene()
hg.LoadSceneFromAssets("studio.scn", scene, res, hg.GetForwardPipelineInfo())

vtx_layout = hg.VertexLayoutPosFloatTexCoord0UInt8()
plane_mdl = hg.CreatePlaneModel(vtx_layout, 1, 1, 1, 1)
plane_prg = hg.LoadProgramFromAssets('shaders/texture')

front_camera = scene.GetNode("front_camera")
side_camera = scene.GetNode("side_camera")
node = scene.GetNode("mixamo_walk_in_circle")

bones = {
    'head': node.GetInstanceSceneView().GetNode(scene, "mixamorig:Head"),
    'right_arm': node.GetInstanceSceneView().GetNode(scene, "mixamorig:RightArm"),
    'left_arm': node.GetInstanceSceneView().GetNode(scene, "mixamorig:LeftArm"),
    'right_leg': node.GetInstanceSceneView().GetNode(scene, "mixamorig:RightLeg"),
    'left_leg': node.GetInstanceSceneView().GetNode(scene, "mixamorig:LeftLeg"),
    'hips': node.GetInstanceSceneView().GetNode(scene, "mixamorig:Hips")
}

frame_buffer, color, tex_color_ref, tex_readback, picture = InitRenderToTexture(res)
frame_buffer_2, color_2, tex_color_ref_2, tex_readback_2, picture_2 = InitRenderToTexture(res, "tex_rb_2", "tex_color_ref_2")

state = state_2 = "none"

while not hg.ReadKeyboard().Key(hg.K_Escape) and hg.IsWindowOpen(win):
    dt = hg.TickClock()
    view_id = 0
    scene.Update(dt)

    for bone_key, bone_node in bones.items():
        real_pos = hg.GetTranslation(bone_node.GetTransform().GetWorld())
        bone_data[bone_key]['real_positions'].append(real_pos)

    scene.SetCurrentCamera(front_camera)
    view_id, pass_ids = hg.SubmitSceneToPipeline(view_id, scene, hg.IntRect(0, 0, 800, 800), True, pipeline, res, frame_buffer.handle)
    scene.SetCurrentCamera(side_camera)
    view_id, pass_ids = hg.SubmitSceneToPipeline(view_id, scene, hg.IntRect(0, 0, 800, 800), True, pipeline, res, frame_buffer_2.handle)

    hg.SetViewPerspective(view_id, 0, 0, res_x, res_y, hg.TranslationMat4(hg.Vec3(0, 0, -0.5)))

    val_uniforms = [hg.MakeUniformSetValue('color', hg.Vec4(1, 1, 1, 1))]
    tex_uniforms = [hg.MakeUniformSetTexture('s_tex', color, 0)]
    tex_uniforms_2 = [hg.MakeUniformSetTexture('s_tex', color_2, 0)]

    hg.DrawModel(view_id, plane_mdl, plane_prg, val_uniforms, tex_uniforms, 
                hg.TransformationMat4(hg.Vec3(-0.25, 0, 0), hg.Vec3(-math.pi / 2, 0.0, 0.0), hg.Vec3(0.40, 0.40, 0.40)))
    hg.DrawModel(view_id, plane_mdl, plane_prg, val_uniforms, tex_uniforms_2, 
                hg.TransformationMat4(hg.Vec3(0.25, 0, 0), hg.Vec3(-math.pi / 2, 0.0, 0.0), hg.Vec3(0.40, 0.40, 0.40)))

    if state == "none":
        state = "capture"
        frame_count_capture, view_id = hg.CaptureTexture(view_id, res, tex_color_ref, tex_readback, picture)
    elif state == "capture" and frame_count_capture <= frame:
        image = GetOpenCvImageFromPicture(picture)
        if image is not None:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            image_with_skeleton = draw_skeleton_on_image(image.copy(), results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Camera Front - MediaPipe Skeleton', image_with_skeleton)
            
            if results.pose_landmarks:
                for bone_key, data in bone_data.items():
                    data['last_front'] = get_landmark_position(results.pose_landmarks, data['mediapipe_landmark'])
        state = "none"

    if state_2 == "none":
        state_2 = "capture"
        frame_count_capture_2, view_id = hg.CaptureTexture(view_id, res, tex_color_ref_2, tex_readback_2, picture_2)
    elif state_2 == "capture" and frame_count_capture_2 <= frame:
        image_2 = GetOpenCvImageFromPicture(picture_2)
        if image_2 is not None:
            results_2 = pose_2.process(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
            
            image_2_with_skeleton = draw_skeleton_on_image(image_2.copy(), results_2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Camera Side - MediaPipe Skeleton', image_2_with_skeleton)
            
            if results_2.pose_landmarks:
                for bone_key, data in bone_data.items():
                    data['last_side'] = get_landmark_position(results_2.pose_landmarks, data['mediapipe_landmark'])
        state_2 = "none"

    for bone_key, data in bone_data.items():
        triangulated_pos = triangulate_position(data['last_front'], data['last_side'])
        if triangulated_pos is not None:
            data['triangulated_positions'].append(triangulated_pos)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame = hg.Frame()
    hg.UpdateWindow(win)

cv2.destroyAllWindows()
pose.close()
pose_2.close()
hg.RenderShutdown()
hg.WindowSystemShutdown()

for bone_key, data in bone_data.items():
    if data['real_positions'] and data['triangulated_positions']:
        optimal_correction = calculate_optimal_correction(data['real_positions'], data['triangulated_positions'])
        
        if optimal_correction is not None:
            data['corrected_positions'] = apply_corrections_to_all_positions(
                data['triangulated_positions'], optimal_correction
            )

global_bounds = calculate_global_bounds(bone_data)

for bone_key, data in bone_data.items():
    if data['real_positions'] and data['triangulated_positions']:
        show_comparison_plot_with_same_scale(
            data['real_positions'], 
            data['triangulated_positions'], 
            data['corrected_positions'], 
            data['name'],
            global_bounds
        )
        
        if data['corrected_positions'] and len(data['corrected_positions']) > 0:
            real_array = np.array([[pos.x, pos.y, pos.z] for pos in data['real_positions'][:len(data['corrected_positions'])]])
            corrected_array = np.array(data['corrected_positions'])
            
            distances = np.linalg.norm(real_array - corrected_array, axis=1)
            mean_distance = np.mean(distances)
            max_distance = np.max(distances)
            std_distance = np.std(distances)
        
        input(f"Appuyez sur Entrée pour continuer après {data['name']}...")
        
        