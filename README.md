# we-video-preview
Wallpaper Engine video preview playback, play Wallpaper Engine videos by calling mpv.<br/>
The main purpose is to quickly (drag the progress bar) view the entire video, facilitate folder classification of videos that are not suitable for display, or quickly unsubscribe from videos that you do not like.<br/>
 again.
The default shortcut key is alt + left click. Since we do not have an API parameter, it will recognize the image pixels and call the right-click to open in File Explorer and play the video inside. If there are recognition issues, you can modify the parameters of pixel recognition yourself, such as threshold, or take a screenshot of the image opened in File Explorer and replace ocr.png.<br/>
Wallpaper Engine视频预览播放，通过调用mpv播放Wallpaper Engine视频。<br/>
主要目的为快速（拖动进度条）查看整个视频，方便对部分不适展示的视频进行文件夹分类，或者自己不喜欢的视频快速取消订阅<br/>
默认快捷键为alt+左键，由于we没有api传参，会通过图片像素识别，调用右键的在资源管理器打开，并播放其中视频。如果识别有问题，可以自行修改像素识别的参数，比如threshold，或者自己截图在资源管理器打开的图片并替换ocr.jpg<br/>
