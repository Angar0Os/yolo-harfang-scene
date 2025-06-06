import harfang as hg
import math
import cv2
import ctypes
import numpy as np
import matplotlib.pyplot as plt 
import mediapipe as mp
import time

def InitRenderToTexture(res, pipeline_texture_name = "tex_rb", texture_name = "tex_color_ref", res_x = 800, res_y = 800):
	frame_buffer = hg.CreateFrameBuffer(res_x, res_y, hg.TF_RGBA8, hg.TF_D24, 4, 'framebuffer')
	color = hg.GetColorTexture(frame_buffer)
	tex_color_ref = res.AddTexture(pipeline_texture_name, color)
	tex_readback = hg.CreateTexture(res_x, res_y, texture_name, hg.TF_ReadBack | hg.TF_BlitDestination, hg.TF_RGBA8)
	picture = hg.Picture(res_x, res_y, hg.PF_RGBA32)
	return frame_buffer, color, tex_color_ref, tex_readback, picture

def InitRenderToTexture2(res, pipeline_texture_name = "tex_rb_2", texture_name = "tex_color_ref_2", res_x = 800, res_y = 800):
	frame_buffer_2 = hg.CreateFrameBuffer(res_x, res_y, hg.TF_RGBA8, hg.TF_D24, 4, 'framebuffer_2')
	color_2 = hg.GetColorTexture(frame_buffer_2)
	tex_color_ref_2 = res.AddTexture(pipeline_texture_name, color_2)
	tex_readback_2 = hg.CreateTexture(res_x, res_y, texture_name, hg.TF_ReadBack | hg.TF_BlitDestination, hg.TF_RGBA8)
	picture_2 = hg.Picture(res_x, res_y, hg.PF_RGBA32)
	return frame_buffer_2, color_2, tex_color_ref_2, tex_readback_2, picture_2
 
def GetOpenCvImageFromPicture(picture):
	picture_width, picture_height = picture.GetWidth(), picture.GetHeight()
	picture_data = picture.GetData()
	bytes_per_pixels = 4
	data_size = picture_width * picture_height * bytes_per_pixels
	buffer = (ctypes.c_char * data_size).from_address(picture_data)
	raw_data = bytes(buffer)
	np_array = np.frombuffer(raw_data, dtype=np.uint8)
	image_rgba = np_array.reshape((picture_height, picture_width, bytes_per_pixels))
	image_bgr = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGR)
	return image_bgr

def draw_pose_landmarks(image, landmarks, connections, highlight_points=None):
	if landmarks is None:
		return image
	
	h, w, _ = image.shape
	
	for connection in connections:
		start_idx = connection[0]
		end_idx = connection[1]
		
		if start_idx < len(landmarks.landmark) and end_idx < len(landmarks.landmark):
			start_point = landmarks.landmark[start_idx]
			end_point = landmarks.landmark[end_idx]
			
			start_x = int(start_point.x * w)
			start_y = int(start_point.y * h)
			end_x = int(end_point.x * w)
			end_y = int(end_point.y * h)
			
			cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
	
	for i, landmark in enumerate(landmarks.landmark):
		x = int(landmark.x * w)
		y = int(landmark.y * h)
		color = (0, 0, 255)  
		
		if highlight_points and i in highlight_points:
			color = (255, 0, 255) 
			cv2.circle(image, (x, y), 8, color, -1)
		else:
			cv2.circle(image, (x, y), 5, color, -1)
	
	return image

def get_landmark_position(landmarks, landmark_idx):
	if landmarks and landmark_idx < len(landmarks.landmark):
		return landmarks.landmark[landmark_idx]
	return None

def triangulate_position(point_front, point_side):
	if point_front is None or point_side is None:
		return None
	
	x = point_front.x - 0.5 
	y = point_front.y - 0.5 
	z = point_side.x - 0.5  
	
	return [x, y, z]

def create_triangulated_view(triangulated_pos, real_pos, title="3D Triangulation View"):
	img_size = 600
	triangulated_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
	triangulated_img[:] = (30, 30, 30)
	
	scale = 150
	center_x, center_y = img_size // 2, img_size // 2
	
	if triangulated_pos is not None:
		tri_x = int(center_x + triangulated_pos[0] * scale)
		tri_y = int(center_y - triangulated_pos[1] * scale) 
		cv2.circle(triangulated_img, (tri_x, tri_y), 8, (0, 0, 255), -1) 
		cv2.putText(triangulated_img, "Triangulated", (tri_x + 15, tri_y), 
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		
		cv2.putText(triangulated_img, f"Triangulated: X={triangulated_pos[0]:.3f}, Y={triangulated_pos[1]:.3f}, Z={triangulated_pos[2]:.3f}", 
					(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
	
	if real_pos is not None:
		real_x = int(center_x + real_pos.x * scale)
		real_y = int(center_y - real_pos.y * scale)
		cv2.circle(triangulated_img, (real_x, real_y), 8, (0, 255, 0), -1)
		cv2.putText(triangulated_img, "Real", (real_x + 15, real_y), 
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
		
		cv2.putText(triangulated_img, f"Real: X={real_pos.x:.3f}, Y={real_pos.y:.3f}, Z={real_pos.z:.3f}", 
					(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
		
		if triangulated_pos is not None:
			error = math.sqrt((triangulated_pos[0] - real_pos.x)**2 + 
							 (triangulated_pos[1] - real_pos.y)**2 + 
							 (triangulated_pos[2] - real_pos.z)**2)
			cv2.putText(triangulated_img, f"Error: {error:.3f}", 
						(10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
			
			cv2.line(triangulated_img, (tri_x, tri_y), (real_x, real_y), (255, 255, 255), 1)
	
	for i in range(0, img_size, 50):
		cv2.line(triangulated_img, (i, 0), (i, img_size), (60, 60, 60), 1)
		cv2.line(triangulated_img, (0, i), (img_size, i), (60, 60, 60), 1)
	
	cv2.line(triangulated_img, (center_x, 0), (center_x, img_size), (100, 100, 100), 2)
	cv2.line(triangulated_img, (0, center_y), (img_size, center_y), (100, 100, 100), 2)
	
	cv2.putText(triangulated_img, title, 
				(10, img_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
	
	return triangulated_img

def show_comparison_plot(real_positions, triangulated_positions, title, bone_name):
	if not real_positions or not triangulated_positions:
		return
		
	fig = plt.figure(figsize=(15, 5))
	fig.suptitle(f'Analyse de {bone_name}', fontsize=16)

	ax1 = fig.add_subplot(131, projection='3d')
	real_x = [pos.x for pos in real_positions]
	real_y = [pos.y for pos in real_positions]
	real_z = [pos.z for pos in real_positions]
	ax1.plot(real_z, real_x, real_y, label="Trajectoire réelle", color='green')
	ax1.set_xlim(-1.5, 1.5)
	ax1.set_ylim(-1.5, 1.5)
	ax1.set_zlim(-1.5, 1.5)
	ax1.set_xlabel('X (avant/arrière)')
	ax1.set_ylabel('Y (gauche/droite)')
	ax1.set_zlabel('Z (haut/bas)')
	ax1.set_title("Trajectoire réelle")
	ax1.legend()
	ax1.view_init(elev=30, azim=45)

	ax2 = fig.add_subplot(132, projection='3d')
	tri_x = [pos[0] for pos in triangulated_positions]
	tri_y = [pos[1] for pos in triangulated_positions]
	tri_z = [pos[2] for pos in triangulated_positions]
	ax2.plot(tri_z, tri_x, tri_y, label="Trajectoire triangulée", color='red')
	ax2.set_xlim(-0.5, 0.5)
	ax2.set_ylim(-0.5, 0.5)
	ax2.set_zlim(-0.5, 0.5)
	ax2.set_xlabel('X (avant/arrière)')
	ax2.set_ylabel('Y (gauche/droite)')
	ax2.set_zlabel('Z (haut/bas)')
	ax2.set_title("Trajectoire triangulée")
	ax2.legend()
	ax2.view_init(elev=30, azim=45)

	ax3 = fig.add_subplot(133, projection='3d')
	ax3.plot(real_z, real_x, real_y, label="Réelle", color='green', alpha=0.7)
	tri_x_scaled = [pos[0] * 3 for pos in triangulated_positions] 
	tri_y_scaled = [pos[1] * 3 for pos in triangulated_positions]
	tri_z_scaled = [pos[2] * 3 for pos in triangulated_positions]
	ax3.plot(tri_z_scaled, tri_x_scaled, tri_y_scaled, label="Triangulée (mise à l'échelle)", color='red', alpha=0.7)
	ax3.set_xlim(-1.5, 1.5)
	ax3.set_ylim(-1.5, 1.5)
	ax3.set_zlim(-1.5, 1.5)
	ax3.set_xlabel('X (avant/arrière)')
	ax3.set_ylabel('Y (gauche/droite)')
	ax3.set_zlabel('Z (haut/bas)')
	ax3.set_title("Comparaison des trajectoires")
	ax3.legend()
	ax3.view_init(elev=30, azim=45)

	plt.tight_layout()
	plt.show()
	
	input(f"Appuyez sur Entrée pour continuer après avoir fermé le graphique de {bone_name}...")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
	static_image_mode=False,
	model_complexity=1,
	smooth_landmarks=True,
	enable_segmentation=False,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5
)

pose_2 = mp_pose.Pose(
	static_image_mode=False,
	model_complexity=1,
	smooth_landmarks=True,
	enable_segmentation=False,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5
)

bone_data = {
	'head': {
		'real_positions': [],
		'triangulated_positions': [],
		'last_front': None,
		'last_side': None,
		'mediapipe_landmark': mp_pose.PoseLandmark.NOSE,
		'name': 'Tête'
	},
	'right_arm': {
		'real_positions': [],
		'triangulated_positions': [],
		'last_front': None,
		'last_side': None,
		'mediapipe_landmark': mp_pose.PoseLandmark.RIGHT_SHOULDER,
		'name': 'Bras Droit'
	},
	'left_arm': {
		'real_positions': [],
		'triangulated_positions': [],
		'last_front': None,
		'last_side': None,
		'mediapipe_landmark': mp_pose.PoseLandmark.LEFT_SHOULDER,
		'name': 'Bras Gauche'
	},
	'right_leg': {
		'real_positions': [],
		'triangulated_positions': [],
		'last_front': None,
		'last_side': None,
		'mediapipe_landmark': mp_pose.PoseLandmark.RIGHT_HIP,
		'name': 'Jambe Droite'
	},
	'left_leg': {
		'real_positions': [],
		'triangulated_positions': [],
		'last_front': None,
		'last_side': None,
		'mediapipe_landmark': mp_pose.PoseLandmark.LEFT_HIP,
		'name': 'Jambe Gauche'
	},
	'hips': {
		'real_positions': [],
		'triangulated_positions': [],
		'last_front': None,
		'last_side': None,
		'mediapipe_landmark': mp_pose.PoseLandmark.RIGHT_HIP, 
		'name': 'Hanches'
	}
}

hg.InputInit()
hg.WindowSystemInit()

res_x, res_y = 1920, 1080
win = hg.RenderInit('Multi-Bone Pose Detection', res_x, res_y, hg.RF_VSync | hg.RF_MSAA8X)

hg.AddAssetsFolder("assets_compiled")

pipeline = hg.CreateForwardPipeline()
res = hg.PipelineResources()
state = "none"
state_2 = "none"
frame = 0

scene = hg.Scene()
hg.LoadSceneFromAssets("studio.scn", scene, res, hg.GetForwardPipelineInfo())

vtx_layout = hg.VertexLayoutPosFloatTexCoord0UInt8()
plane_mdl = hg.CreatePlaneModel(vtx_layout, 1, 1, 1, 1)
plane_ref = res.AddModel('plane', plane_mdl)
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
frame_buffer_2, color_2, tex_color_ref_2, tex_readback_2, picture_2 = InitRenderToTexture2(res)

cv2.namedWindow('Pose Detection - Front Camera', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Pose Detection - Side Camera', cv2.WINDOW_AUTOSIZE)

triangulation_windows = {}
for bone_key in bone_data.keys():
	window_name = f'3D Triangulation - {bone_data[bone_key]["name"]}'
	cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
	triangulation_windows[bone_key] = window_name

print("Début de la capture. Appuyez sur 'Esc' pour arrêter et voir les analyses.")

while not hg.ReadKeyboard().Key(hg.K_Escape) and hg.IsWindowOpen(win):
	dt = hg.TickClock()
	view_id = 0
	scene.Update(dt)

	real_positions = {}
	for bone_key, bone_node in bones.items():
		real_positions[bone_key] = hg.GetTranslation(bone_node.GetTransform().GetWorld())
		bone_data[bone_key]['real_positions'].append(real_positions[bone_key])

	scene.SetCurrentCamera(front_camera)
	view_id, pass_ids = hg.SubmitSceneToPipeline(view_id, scene, hg.IntRect(0, 0, 800, 800), True, pipeline, res, frame_buffer.handle)
	scene.SetCurrentCamera(side_camera)
	view_id, pass_ids = hg.SubmitSceneToPipeline(view_id, scene, hg.IntRect(0, 0, 800, 800), True, pipeline, res, frame_buffer_2.handle) 	

	hg.SetViewPerspective(view_id, 0, 0, res_x, res_y, hg.TranslationMat4(hg.Vec3(0, 0, -0.5)))

	val_uniforms = [hg.MakeUniformSetValue('color', hg.Vec4(1, 1, 1, 1))] 
	tex_uniforms = [hg.MakeUniformSetTexture('s_tex', color, 0)]
	tex_uniforms_2 = [hg.MakeUniformSetTexture('s_tex', color_2, 0)]

	hg.DrawModel(view_id, plane_mdl, plane_prg, val_uniforms, tex_uniforms, hg.TransformationMat4(hg.Vec3(-0.25, 0, 0),  hg.Vec3(-math.pi / 2, 0.0, 0.0),  hg.Vec3(0.40, 0.40, 0.40)))
	hg.DrawModel(view_id, plane_mdl, plane_prg, val_uniforms, tex_uniforms_2, hg.TransformationMat4(hg.Vec3(0.25, 0, 0),  hg.Vec3(-math.pi / 2, 0.0, 0.0),  hg.Vec3(0.40, 0.40, 0.40)))

	if state == "none":
		state = "capture"
		frame_count_capture, view_id = hg.CaptureTexture(view_id, res, tex_color_ref, tex_readback, picture)
	elif state == "capture" and frame_count_capture <= frame:
		image = GetOpenCvImageFromPicture(picture)
		
		if image is not None:
			image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			results = pose.process(image_rgb)
			display_image = image.copy()
			
			if results.pose_landmarks:
				highlight_points = [data['mediapipe_landmark'].value for data in bone_data.values()]
				display_image = draw_pose_landmarks(display_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, highlight_points)
				
				for bone_key, data in bone_data.items():
					landmark = get_landmark_position(results.pose_landmarks, data['mediapipe_landmark'])
					data['last_front'] = landmark
					
					if landmark:
						cv2.putText(display_image, 
								   f"{data['name']}: ({landmark.x:.2f}, {landmark.y:.2f})", 
								   (10, 30 + list(bone_data.keys()).index(bone_key) * 25), 
								   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
			
			cv2.putText(display_image, "Front Camera", (10, display_image.shape[0] - 10), 
					   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
			cv2.imshow('Pose Detection - Front Camera', display_image)
			state = "none"

	if state_2 == "none":
		state_2 = "capture"
		frame_count_capture_2, view_id = hg.CaptureTexture(view_id, res, tex_color_ref_2, tex_readback_2, picture_2)
	elif state_2 == "capture" and frame_count_capture_2 <= frame:
		image_2 = GetOpenCvImageFromPicture(picture_2)
		
		if image_2 is not None:
			image_rgb_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
			results_2 = pose_2.process(image_rgb_2)
			display_image_2 = image_2.copy()
			
			if results_2.pose_landmarks:
				highlight_points = [data['mediapipe_landmark'].value for data in bone_data.values()]
				display_image_2 = draw_pose_landmarks(display_image_2, results_2.pose_landmarks, mp_pose.POSE_CONNECTIONS, highlight_points)
				
				for bone_key, data in bone_data.items():
					landmark = get_landmark_position(results_2.pose_landmarks, data['mediapipe_landmark'])
					data['last_side'] = landmark
					
					if landmark:
						cv2.putText(display_image_2, 
								   f"{data['name']}: ({landmark.x:.2f}, {landmark.y:.2f})", 
								   (10, 30 + list(bone_data.keys()).index(bone_key) * 25), 
								   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
			
			cv2.putText(display_image_2, "Side Camera", (10, display_image_2.shape[0] - 10), 
					   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
			cv2.imshow('Pose Detection - Side Camera', display_image_2)
			state_2 = "none"
	
	for bone_key, data in bone_data.items():
		triangulated_pos = triangulate_position(data['last_front'], data['last_side'])
		if triangulated_pos is not None:
			data['triangulated_positions'].append(triangulated_pos)
		
		triangulated_view = create_triangulated_view(
			triangulated_pos, 
			real_positions[bone_key], 
			f"3D Triangulation - {data['name']}"
		)
		cv2.imshow(triangulation_windows[bone_key], triangulated_view)
   
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	frame = hg.Frame()
	hg.UpdateWindow(win)

cv2.destroyAllWindows()
pose.close()
pose_2.close()

hg.RenderShutdown()
hg.WindowSystemShutdown()

print("\nAnalyse terminée. Affichage des comparaisons successives...")

for bone_key, data in bone_data.items():
	if data['real_positions'] and data['triangulated_positions']:
		print(f"\nAffichage de l'analyse pour : {data['name']}")
		show_comparison_plot(
			data['real_positions'], 
			data['triangulated_positions'], 
			f"Analyse de {data['name']}", 
			data['name']
		)
	else:
		print(f"Pas assez de données pour {data['name']}")

print("\nAnalyse complète terminée !")