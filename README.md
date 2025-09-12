# we-video-preview
Wallpaper Engine video preview playback, playing Wallpaper Engine videos by calling mpv.<br/>
The main purpose is to quickly (drag the progress bar) view the entire video, making it easy to categorize videos that are not suitable for display in folders, or quickly unsubscribe from videos you don't like.<br/>
The default shortcut key is alt + left click. Since we does not have API parameters, it will recognize pixels through images and call right-click to open in File Explorer and play the video. If there are recognition issues, you can modify the pixel recognition parameters yourself, such as threshold, or take a screenshot of the image opened in File Explorer and replace ocr.jpg<br/>
You need to install mpv or ffplay. The config.toml file needs to write down the path of mpv or ffplay. ffplay does not support dragging progress bars and can only be fast-forwarded or rewound using arrow keys.<br/>
Wallpaper Engine视频预览播放，通过调用mpv播放Wallpaper Engine视频。<br/>
主要目的为快速（拖动进度条）查看整个视频，方便对部分不适展示的视频进行文件夹分类，或者自己不喜欢的视频快速取消订阅<br/>
默认快捷键为alt+左键，由于we没有api传参，会通过图片像素识别，调用右键的在资源管理器打开，并播放其中视频。如果识别有问题，可以自行修改像素识别的参数，比如threshold，或者自己截图在资源管理器打开的图片并替换ocr.jpg<br/>
需要安装mpv或者ffplay，config.toml文件内需要写入mpv或者ffplay的路径，ffplay不支持进度条拖动，只能用方向键快进快退<br/>
