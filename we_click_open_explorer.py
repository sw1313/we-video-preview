# we_click_open_explorer.py
import os
import time
from datetime import datetime
import threading
import subprocess
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui as pag
from PIL import Image

import keyboard
import mouse
from mouse import ButtonEvent

# ==========================
# 外置配置加载（config.toml / config.json）
# ==========================
import json
import copy

BASE_DIR = Path(__file__).resolve().parent

DEFAULTS = {
    "players": {
        "mpv_candidates": [
            r"C:\Program Files\mpv\mpv.exe",
            r"C:\Program Files (x86)\mpv\mpv.exe",
            r"C:\Users\Public\scoop\apps\mpv\current\mpv.exe",
        ],
        "ffplay_candidates": [
            r"C:\Windows\System32\ffplay.exe",
            r"C:\ffmpeg\bin\ffplay.exe",
        ],
        "mpv": {
            "args_new": [
                "--vo=gpu-next",
                "--tone-mapping=clip",
                "--target-peak=500",
                "--ontop",
                "--keep-open=no",
                "--osd-level=1",
                "--no-terminal",
            ],
            "args_old": [
                "--vo=gpu",
                "--tone-mapping=clip",
                "--target-peak=500",
                "--ontop",
                "--keep-open=no",
                "--osd-level=1",
                "--no-terminal",
            ],
        },
        "ffplay": {
            "args": ["-autoexit", "-hide_banner", "-loglevel", "error"],
        },
    },
    "match": {
        "template": "ocr.png",
        "search_whole_screen": False,
        "search_box_w": 1000,
        "search_box_h": 1200,
        "scales": [1.00, 1.10, 0.90, 1.15, 0.85, 1.20, 0.95, 1.05, 1.30, 0.80],
        "threshold": 0.80,
        "context_menu_delay": 0.15,
        "save_debug_image": True,
        "debug_dir": "debug",
        "debounce_sec": 0.35,
    },
    "video": {
        "exts": [".mp4", ".mkv", ".webm", ".mov", ".avi"],
        "max_depth": 3,
    },
    "pyautogui": {
        "failsafe": True,
        "pause": 0.02,
    },
    "hotkeys": {
        "trigger_mods": ["alt"],   # 可多项: ctrl/shift/alt/win
        "trigger_button": "left",  # left/right/middle
        "exit": "ctrl+alt+q",
    },
}

def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_toml(path: Path) -> dict:
    try:
        try:
            import tomllib  # py3.11+
        except ModuleNotFoundError:
            import tomli as tomllib  # py3.10-
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def load_config() -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    loaded = {}
    p_toml = BASE_DIR / "config.toml"
    p_json = BASE_DIR / "config.json"
    if p_toml.exists():
        loaded = _load_toml(p_toml)
    elif p_json.exists():
        loaded = _load_json(p_json)
    if loaded:
        _deep_update(cfg, loaded)
    return cfg

CFG = load_config()

def _resolve_path(p: str) -> str:
    if not p:
        return p
    pp = Path(p)
    return str(pp if pp.is_absolute() else (BASE_DIR / pp))

# ---- 展开配置 ----
MPV_CANDIDATES = [_resolve_path(p) for p in CFG["players"]["mpv_candidates"]]
FFPLAY_CANDIDATES = [_resolve_path(p) for p in CFG["players"]["ffplay_candidates"]]

TEMPLATE_PATH       = _resolve_path(CFG["match"]["template"])
SEARCH_WHOLE_SCREEN = bool(CFG["match"]["search_whole_screen"])
SEARCH_BOX_W        = int(CFG["match"]["search_box_w"])
SEARCH_BOX_H        = int(CFG["match"]["search_box_h"])
SCALES              = [float(x) for x in CFG["match"]["scales"]]
THRESHOLD           = float(CFG["match"]["threshold"])
CONTEXT_MENU_DELAY  = float(CFG["match"]["context_menu_delay"])
SAVE_DEBUG_IMAGE    = bool(CFG["match"]["save_debug_image"])
DEBUG_DIR           = _resolve_path(CFG["match"]["debug_dir"])
DEBOUNCE_SEC        = float(CFG["match"]["debounce_sec"])

VIDEO_EXTS = [str(x).lower() for x in CFG["video"]["exts"]]
MAX_DEPTH  = int(CFG["video"]["max_depth"])

FFPLAY_ARGS  = [str(x) for x in CFG["players"]["ffplay"]["args"]]
MPV_ARGS_NEW = [str(x) for x in CFG["players"]["mpv"]["args_new"]]
MPV_ARGS_OLD = [str(x) for x in CFG["players"]["mpv"]["args_old"]]

TRIGGER_MODS   = [s.lower() for s in CFG["hotkeys"]["trigger_mods"]]
TRIGGER_BUTTON = str(CFG["hotkeys"]["trigger_button"]).lower()
EXIT_HOTKEY    = str(CFG["hotkeys"]["exit"])

# PyAutoGUI
pag.FAILSAFE = bool(CFG["pyautogui"]["failsafe"])
pag.PAUSE    = float(CFG["pyautogui"]["pause"])

# ====== 播放器探测 ======
def find_exe(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    from shutil import which
    return which(os.path.basename(candidates[0])) or None

PLAYER_MPV = find_exe(MPV_CANDIDATES)
PLAYER_FFPLAY = find_exe(FFPLAY_CANDIDATES)

def _try_spawn(cmd):
    try:
        p = subprocess.Popen(cmd)
    except Exception as e:
        print("[PLAYER ERROR]", e)
        return False
    time.sleep(1.0)
    return p.poll() is None

def play_video(path):
    if not os.path.exists(path):
        print(f"[WARN] 视频不存在：{path}")
        return
    if PLAYER_MPV:
        cmd_new = [PLAYER_MPV, *MPV_ARGS_NEW, path]
        if _try_spawn(cmd_new):
            print(f"[PLAY] mpv (gpu-next / SDR sRGB)：{path}")
            return
        print("[INFO] mpv 新参数不被当前版本接受，尝试旧参数...")
        cmd_old = [PLAYER_MPV, *MPV_ARGS_OLD, path]
        if _try_spawn(cmd_old):
            print(f"[PLAY] mpv (gpu / SDR BT.709)：{path}")
            return
        print("[WARN] mpv 也无法使用提供的参数，回退到 ffplay。")
    if PLAYER_FFPLAY:
        cmd = [PLAYER_FFPLAY, *FFPLAY_ARGS, "-i", path]
        if _try_spawn(cmd):
            print(f"[PLAY] ffplay（无进度条）：{path}")
            return
        print("[ERROR] ffplay 启动失败。")
    else:
        print("[ERROR] 没找到可用播放器。建议安装 mpv。")

# ===== Explorer / 窗口工具（pywin32）=====
import win32com.client
import win32gui
import win32con

def get_explorer_hwnds():
    shell = win32com.client.Dispatch("Shell.Application")
    res = {}
    for w in shell.Windows():
        try:
            if w and w.FullName and str(w.FullName).lower().endswith("explorer.exe"):
                hwnd = int(w.HWND)
                path = ""
                try:
                    path = str(w.Document.Folder.Self.Path)
                except Exception:
                    path = ""
                res[hwnd] = path
        except Exception:
            pass
    return res

def wait_new_explorer(before_map, timeout_sec=8.0, poll=0.05):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        now = get_explorer_hwnds()
        for hwnd, path in now.items():
            if hwnd not in before_map:
                return hwnd, path or ""
        time.sleep(poll)
    return None, ""

def minimize_explorer(hwnd):
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_SHOWMINIMIZED)
        SWP_FLAGS = (win32con.SWP_NOMOVE | win32con.SWP_NOSIZE |
                     win32con.SWP_NOACTIVATE | win32con.SWP_NOZORDER)
        win32gui.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_FLAGS)
    except Exception:
        pass

def close_explorer(hwnd, timeout_sec=2.0):
    """关闭指定的资源管理器窗口；若未能及时关闭则降级隐藏。"""
    try:
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if not win32gui.IsWindow(hwnd):
                return
            time.sleep(0.05)
        # 若仍存在，降级隐藏（确保不打扰前台）
        win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
    except Exception:
        pass

# ===== 文件/视频工具 =====
def find_video(root: str, exts=VIDEO_EXTS, max_depth=0):
    root_p = Path(root)
    if not root_p.exists():
        return None
    for p in root_p.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            return str(p)
    if max_depth > 0:
        for p in root_p.iterdir():
            if p.is_dir():
                name = p.name.lower()
                if name in ("preview", "thumb", "cover", "assets", "__pycache__"):
                    continue
                hit = find_video(str(p), exts, max_depth - 1)
                if hit:
                    return hit
    return None

# ===== 图像/截屏/匹配 =====
def grab_region(left, top, width, height):
    with mss.mss() as sct:
        mon = sct.monitors[0]
        l = max(0, int(left))
        t = max(0, int(top))
        r = min(mon["width"], int(left + width))
        b = min(mon["height"], int(top + height))
        w = max(1, r - l)
        h = max(1, b - t)
        raw = sct.grab({"left": l, "top": t, "width": w, "height": h})
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), l, t

def load_template_gray(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"模板不存在：{path}")
    tpl = cv2.imread(path, cv2.IMREAD_COLOR)
    if tpl is None:
        raise RuntimeError(f"无法读取模板图像：{path}")
    return cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

def match_template_multi(image_bgr, tpl_gray, scales, threshold):
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img_gray.shape[:2]
    best_val, best_center, best_rect, best_scale = -1.0, None, None, None
    th0, tw0 = tpl_gray.shape[:2]
    for s in scales:
        new_w = max(1, int(tw0 * s))
        new_h = max(1, int(th0 * s))
        if new_w > w_img or new_h > h_img:
            continue
        tpl_s = cv2.resize(tpl_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        res = cv2.matchTemplate(img_gray, tpl_s, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            x1, y1 = max_loc
            x2, y2 = x1 + new_w, y1 + new_h
            best_center = (x1 + new_w // 2, y1 + new_h // 2)
            best_rect = ((x1, y1), (x2, y2))
            best_scale = s
    if best_val >= threshold and best_center is not None:
        return True, best_center[0], best_center[1], best_val, best_scale, best_rect
    else:
        return False, -1, -1, best_val, best_scale, best_rect

def save_debug(image_bgr, rect, score, scale, left, top, label="hit"):
    if not SAVE_DEBUG_IMAGE:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    img = image_bgr.copy()
    if rect is not None:
        (x1, y1), (x2, y2) = rect
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        sc = f"{score:.3f}" if isinstance(score, float) else str(score)
        sca = f"{scale:.2f}" if isinstance(scale, (float, int)) else "n/a"
        cv2.putText(img, f"{label} s={sca} sc={sc}",
                    (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{label}_{ts}_{left}x{top}.png"), img)

# ===== 主动作：右键→截屏→命中→Explorer→关闭→播放 =====
_trigger_lock = threading.Lock()
_last_trigger = 0.0

def right_click_and_find():
    try:
        before = get_explorer_hwnds()
        mx, my = pag.position()
        pag.click(mx, my, button="right")
        time.sleep(CONTEXT_MENU_DELAY)

        if SEARCH_WHOLE_SCREEN:
            screen_w, screen_h = pag.size()
            shot_bgr, left, top = grab_region(0, 0, screen_w, screen_h)
        else:
            shot_bgr, left, top = grab_region(mx, my, SEARCH_BOX_W, SEARCH_BOX_H)

        tpl_gray = load_template_gray(TEMPLATE_PATH)
        ok, cx, cy, score, scale, rect = match_template_multi(shot_bgr, tpl_gray, SCALES, THRESHOLD)

        if not ok:
            save_debug(shot_bgr, rect, score, scale, left, top, "miss")
            print(f"[MISS] 未命中，best_score={score if not isinstance(score,float) else round(score,3)}  scale={scale}")
            return

        gx, gy = left + cx, top + cy
        save_debug(shot_bgr, rect, score, scale, left, top, "hit")
        pag.click(gx, gy, button="left")
        print(f"[OK] 命中 @ ({gx},{gy})  score={score:.3f}  scale={scale:.2f}")

        hwnd, path = wait_new_explorer(before, timeout_sec=8.0, poll=0.05)
        if not hwnd:
            print("[WARN] 未检测到新打开的资源管理器窗口。")
            return

        # 先补齐路径
        if not path:
            time.sleep(0.2)
            cur = get_explorer_hwnds()
            path = cur.get(hwnd, "")
        print(f"[EXPLORER] hwnd={hwnd}, path={path}")

        # 立刻关闭该 Explorer 窗口，避免干扰
        close_explorer(hwnd)

        if not path or not os.path.exists(path):
            print("[WARN] 未获取到有效目录路径。")
            return

        video = find_video(path, VIDEO_EXTS, MAX_DEPTH)
        if not video:
            print(f"[INFO] 目录中未找到常见视频：{path}")
            return

        play_video(video)

    except Exception as e:
        print("[ERROR]", e)

# ===== 仅在按住修饰键时装鼠标钩子 =====
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

WH_MOUSE_LL = 14
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP   = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP   = 0x0205
WM_MBUTTONDOWN = 0x0207
WM_MBUTTONUP   = 0x0208
WM_QUIT        = 0x0012

VK_SHIFT   = 0x10
VK_CONTROL = 0x11
VK_MENU    = 0x12  # Alt
VK_LWIN    = 0x5B
VK_RWIN    = 0x5C

LLMHF_INJECTED          = 0x00000001
LLMHF_LOWER_IL_INJECTED = 0x00000002

MOD2VK = {
    "shift":   VK_SHIFT,
    "ctrl":    VK_CONTROL,
    "control": VK_CONTROL,
    "alt":     VK_MENU,
    "win":     VK_LWIN,
}

BUTTON_WM = {
    "left":  (WM_LBUTTONDOWN, WM_LBUTTONUP),
    "right": (WM_RBUTTONDOWN, WM_RBUTTONUP),
    "middle":(WM_MBUTTONDOWN, WM_MBUTTONUP),
}

# 32/64 位兼容类型
if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
    LONG_PTR  = ctypes.c_longlong
else:
    ULONG_PTR = ctypes.c_ulong
    LONG_PTR  = ctypes.c_long
LRESULT = LONG_PTR

class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ('pt', wintypes.POINT),
        ('mouseData', wintypes.DWORD),
        ('flags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', ULONG_PTR),
    ]

LowLevelMouseProc = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

# 明确签名
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, LowLevelMouseProc, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype  = ctypes.c_void_p
user32.CallNextHookEx.argtypes    = [ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype     = LRESULT
user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
user32.UnhookWindowsHookEx.restype  = wintypes.BOOL
kernel32.GetModuleHandleW.argtypes  = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype   = wintypes.HMODULE
user32.PostThreadMessageW.argtypes  = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostThreadMessageW.restype   = wintypes.BOOL
kernel32.GetCurrentThreadId.restype = wintypes.DWORD

HOOK_HANDLE = None
HOOK_THREAD = None
HOOK_THREAD_ID = None
STOP_HOOK   = threading.Event()

# 状态：仅在“本次按住修饰键”期间吞掉第一次触发键（down+up）
hotkey_used_this_hold = False
swallow_pair_active = False

def _mods_pressed_win():
    for m in TRIGGER_MODS:
        m = m.lower()
        if m == "win":
            if not (user32.GetAsyncKeyState(VK_LWIN) & 0x8000 or user32.GetAsyncKeyState(VK_RWIN) & 0x8000):
                return False
        else:
            vk = MOD2VK.get(m)
            if vk is None:
                return False
            if (user32.GetAsyncKeyState(vk) & 0x8000) == 0:
                return False
    return True

def _button_pair_for_trigger():
    return BUTTON_WM.get(TRIGGER_BUTTON, BUTTON_WM["left"])

@LowLevelMouseProc
def _mouse_proc(nCode, wParam, lParam):
    global hotkey_used_this_hold, swallow_pair_active, _last_trigger
    if nCode != 0:
        return user32.CallNextHookEx(HOOK_HANDLE, nCode, wParam, lParam)

    hs = ctypes.cast(lParam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents

    # 忽略注入事件（避免我们模拟的点击被钩子吞掉）
    if hs.flags & (LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED):
        return user32.CallNextHookEx(HOOK_HANDLE, nCode, wParam, lParam)

    downs, ups = _button_pair_for_trigger()

    # 正在吞这对按键：继续吞到对应的 up 为止
    if swallow_pair_active:
        if wParam == ups:
            swallow_pair_active = False
        return 1

    # 只有修饰键按住时才判断/触发；否则立即放行
    if not _mods_pressed_win():
        return user32.CallNextHookEx(HOOK_HANDLE, nCode, wParam, lParam)

    # 仅在本次按住期间第一次触发
    if (not hotkey_used_this_hold) and (wParam == downs):
        swallow_pair_active = True
        hotkey_used_this_hold = True
        now = time.time()
        with _trigger_lock:
            if now - _last_trigger >= DEBOUNCE_SEC:
                _last_trigger = now
                threading.Thread(target=right_click_and_find, daemon=True).start()
        return 1

    return user32.CallNextHookEx(HOOK_HANDLE, nCode, wParam, lParam)

def _hook_loop():
    global HOOK_HANDLE, HOOK_THREAD_ID
    HOOK_THREAD_ID = kernel32.GetCurrentThreadId()
    hInstance = kernel32.GetModuleHandleW(None)
    HOOK_HANDLE = user32.SetWindowsHookExW(WH_MOUSE_LL, _mouse_proc, hInstance, 0)
    if not HOOK_HANDLE:
        return
    msg = wintypes.MSG()
    # 阻塞式消息泵，降低 CPU 占用
    while not STOP_HOOK.is_set() and user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) != 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))
    if HOOK_HANDLE:
        user32.UnhookWindowsHookEx(HOOK_HANDLE)
        HOOK_HANDLE = None
    HOOK_THREAD_ID = None

def _ensure_hook_started():
    """按下修饰键时调用：若未装钩子则装，重置本次按住的状态。"""
    global HOOK_THREAD, hotkey_used_this_hold, swallow_pair_active
    if HOOK_THREAD and HOOK_THREAD.is_alive():
        return
    hotkey_used_this_hold = False
    swallow_pair_active = False
    STOP_HOOK.clear()
    HOOK_THREAD = threading.Thread(target=_hook_loop, daemon=True, name="LLMouseHook")
    HOOK_THREAD.start()

def _ensure_hook_stopped():
    """松开任一修饰键时调用：卸钩子，彻底停止后台处理。"""
    global hotkey_used_this_hold, swallow_pair_active
    hotkey_used_this_hold = False
    swallow_pair_active = False
    if HOOK_THREAD and HOOK_THREAD.is_alive():
        STOP_HOOK.set()
        if HOOK_THREAD_ID:
            user32.PostThreadMessageW(HOOK_THREAD_ID, WM_QUIT, 0, 0)

# —— 键盘事件：按下任一触发修饰键 → 安装钩子；松开任一 → 卸载钩子 ——
def _norm_mod(m): return "windows" if m.lower()=="win" else m.lower()

def _is_trigger_mod(key_name: str) -> bool:
    name = key_name.lower()
    if "windows" in name or name in ("left windows","right windows","win","left win","right win"):
        return "win" in TRIGGER_MODS or "windows" in TRIGGER_MODS
    return any(name == _norm_mod(m) for m in TRIGGER_MODS)

def _on_key_event(e):
    try:
        if not hasattr(e, "name") or e.name is None:
            return
        if not _is_trigger_mod(e.name):
            return
        if e.event_type == "down":
            # 只有当“全部修饰键”都按下时，才启动钩子
            if all(keyboard.is_pressed(_norm_mod(m)) for m in TRIGGER_MODS):
                _ensure_hook_started()
        elif e.event_type == "up":
            _ensure_hook_stopped()
    except Exception:
        pass

def main():
    print("启动成功：")
    print(f"  - 触发：{' + '.join([*TRIGGER_MODS, TRIGGER_BUTTON])}（仅在按住修饰键时短暂启用鼠标钩子）")
    print(f"  - 退出：{EXIT_HOTKEY}")
    if not os.path.exists(TEMPLATE_PATH):
        print(f"[提示] 模板未找到：{TEMPLATE_PATH}")
    if PLAYER_MPV:
        print(f"[播放器] mpv：{PLAYER_MPV}")
    elif PLAYER_FFPLAY:
        print(f"[播放器] ffplay：{PLAYER_FFPLAY}（无进度条）")
    else:
        print("[播放器] 未找到 mpv/ffplay，请安装 mpv 或 FFmpeg。")

    # 仅监听键盘修饰键的按/抬；需要时才装鼠标钩子
    keyboard.hook(_on_key_event)

    try:
        keyboard.wait(EXIT_HOTKEY)
    finally:
        _ensure_hook_stopped()
        keyboard.unhook(_on_key_event)
        print("[状态] 已退出。")

if __name__ == "__main__":
    main()