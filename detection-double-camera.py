import harfang as hg
import math
import cv2
import ctypes
import numpy as np
import matplotlib.pyplot as plt 

def InitRenderToTexture(res, pipeline_texture_name = "tex_rb", texture_name = "tex_color_ref", res_x = 800, res_y = 800):
	frame_buffer = hg.CreateFrameBuffer(res_x, res_y, hg.TF_RGBA8, hg.TF_D24, 4, 'framebuffer')
	frame_buffer_2 = hg.CreateFrameBuffer(res_x, res_y, hg.TF_RGBA32F, hg.TF_D24, 4, 'framebuffer_2')
	  
	color = hg.GetColorTexture(frame_buffer)
	color_2 = hg.GetColorTexture(frame_buffer_2)
 
	tex_color_ref = res.AddTexture(pipeline_texture_name, color)
	tex_readback = hg.CreateTexture(res_x, res_y, texture_name, hg.TF_ReadBack | hg.TF_BlitDestination, hg.TF_RGBA8)

	picture = hg.Picture(res_x, res_y, hg.PF_RGBA32)

	return frame_buffer, frame_buffer_2, color, color_2, tex_color_ref, tex_readback, picture

def GetOpenCvImageFromPicture(picture):
	picture_width, picture_height = picture.GetWidth(), picture.GetHeight()
	picture_data = picture.GetData()
	bytes_per_pixels = 4
	data_size = picture_width * picture_height * bytes_per_pixels
	buffer = (ctypes.c_char * data_size).from_address(picture_data)
	raw_data = bytes(buffer)
	np_array = np.frombuffer(raw_data, dtype=np.uint8)
	image_rgba = np_array.reshape((picture_height, picture_width, bytes_per_pixels))
	image_bgr = cv2.cvtColor(image_rgba, cv2.COLOR_BGR2RGB)

	return image_bgr

head_positions_x = []
head_positions_y = []
head_positions_z = []

hg.InputInit()
hg.WindowSystemInit()

res_x, res_y = 1920, 1080
win = hg.RenderInit('Draw Scene to Texture', res_x, res_y, hg.RF_VSync | hg.RF_MSAA8X)

hg.AddAssetsFolder("assets_compiled")

pipeline = hg.CreateForwardPipeline()
res = hg.PipelineResources()
state = "none"
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
head = node.GetInstanceSceneView().GetNode(scene, "mixamorig:Head")

frame_buffer, frame_buffer_2, color, color_2, tex_color_ref, tex_readback, picture = InitRenderToTexture(res) 

while not hg.ReadKeyboard().Key(hg.K_Escape) and hg.IsWindowOpen(win):
	dt = hg.TickClock()
	# dt = hg.time_from_sec_f(1/60)
	view_id = 0
	scene.Update(dt)

	head_pos = hg.GetTranslation(head.GetTransform().GetWorld())

	print(head_pos.x)
	print(head_pos.y)
	print(head_pos.z)

	head_positions_x.append(head_pos.x)
	head_positions_y.append(head_pos.y)
	head_positions_z.append(head_pos.z)

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
		# hg.SavePNG(picture, "frame" + str(frame) + ".png")
		if image is not None:
			state = "none"

	
	frame = hg.Frame()
	hg.UpdateWindow(win)

hg.RenderShutdown()
hg.WindowSystemShutdown()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(head_positions_z, head_positions_x, head_positions_y, label="Trajectoire de la tête")

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

ax.set_ylabel('Y (gauche/droite)')
ax.set_zlabel('Z (haut/bas)')
ax.set_xlabel('X (avant/arrière)')

ax.set_title("Tracé de la tête du personnage (3m x 3m)")
ax.legend()
ax.view_init(elev=30, azim=45)  

plt.tight_layout()
plt.show()
	
