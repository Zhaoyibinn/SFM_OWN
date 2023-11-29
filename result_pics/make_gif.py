import os
import imageio

# 1. 指定文件夹和文件名模板
folder_path = f'../test0123'  # 替换为实际的文件夹路径
output_gif_filename = folder_path+'.gif'
output_mp4_filename = folder_path+'.mp4'

# 2. 获取文件夹中的所有PNG图像文件
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
image_files.sort()
frames = []
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    frame = imageio.imread(image_path)
    frames.append(frame)

# 5. 保存为GIF文件
imageio.mimsave(output_gif_filename, frames, duration=0.1)  # 设置帧之间的持续时间

# 6. 保存为MP4文件
# imageio.mimsave(output_mp4_filename, frames, 'mp4')

print(f'GIF文件已保存为: {output_gif_filename}')
print(f'MP4文件已保存为: {output_mp4_filename}')
