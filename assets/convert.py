from moviepy.editor import VideoFileClip

# Load the video file
video_path = 'animation_v4.mp4'
video = VideoFileClip(video_path)

# Adjust frame rate and resize the video
modified_video = video.resize(0.5)

# Convert the modified video to GIF
gif_path = 'animation_v4.gif'
modified_video.write_gif(gif_path, fps=30)  # Ensure GIF also uses 10 fps