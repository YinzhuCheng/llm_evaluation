import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple


PROFILES_PATH = os.path.join(os.path.dirname(__file__), "gui_profiles.json")
EVAL_SCRIPT = os.path.join(os.path.dirname(__file__), "eval_questions.py")


def _safe_read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _safe_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _is_blank(s: Any) -> bool:
    if s is None:
        return True
    s2 = str(s).strip()
    return not s2


def _int_or_none(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _float_or_none(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _build_arg_list(cfg: Dict[str, Any]) -> List[str]:
    """Return argv for eval_questions.py (excluding python executable)."""

    def add(flag: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str) and _is_blank(value):
            return
        args.extend([flag, str(value)])

    def add_bool(flag: str, enabled: bool) -> None:
        if enabled:
            args.append(flag)

    args: List[str] = [EVAL_SCRIPT]

    # Core
    add("--config", cfg.get("config"))
    add("--input", cfg.get("input"))
    add("--sheet", cfg.get("sheet"))
    add("--images-root", cfg.get("images_root"))
    add("--out-dir", cfg.get("out_dir"))

    # Concurrency / retry
    add("--concurrency", cfg.get("concurrency"))
    add("--model-concurrency", cfg.get("model_concurrency"))
    add("--judge-concurrency", cfg.get("judge_concurrency"))
    add("--max-retries", cfg.get("max_retries"))
    add("--retry-base-delay-s", cfg.get("retry_base_delay_s"))
    add("--retry-max-delay-s", cfg.get("retry_max_delay_s"))

    # Eval controls
    add("--limit", cfg.get("limit"))
    add("--cot", cfg.get("cot"))
    add("--answer-json-max-attempts", cfg.get("answer_json_max_attempts"))

    # Network
    add("--vpn", cfg.get("vpn"))
    add("--proxy", cfg.get("proxy"))

    # Model
    add("--model-provider", cfg.get("model_provider"))
    add("--model-base-url", cfg.get("model_base_url"))
    add("--model-api-key", cfg.get("model_api_key"))
    add("--model-name", cfg.get("model_name"))
    add("--model-timeout-s", cfg.get("model_timeout_s"))
    add("--model-temperature", cfg.get("model_temperature"))
    add("--model-max-tokens", cfg.get("model_max_tokens"))

    # Judge
    add("--judge-provider", cfg.get("judge_provider"))
    add("--judge-base-url", cfg.get("judge_base_url"))
    add("--judge-api-key", cfg.get("judge_api_key"))
    add("--judge-name", cfg.get("judge_name"))
    add("--judge-timeout-s", cfg.get("judge_timeout_s"))
    add("--judge-temperature", cfg.get("judge_temperature"))
    add("--judge-max-tokens", cfg.get("judge_max_tokens"))

    return args


def _format_cmd_for_display(argv: List[str]) -> str:
    """Human-friendly command string; uses Windows quoting rules on Windows."""
    full = [sys.executable] + argv
    if os.name == "nt":
        # Windows-style quoting
        return subprocess.list2cmdline(full)
    # POSIX-ish quoting
    def q(a: str) -> str:
        if not a:
            return "''"
        if any(ch.isspace() for ch in a) or any(ch in a for ch in ['"', "'", "\\"]):
            return "'" + a.replace("'", "'\\''") + "'"
        return a

    return " ".join(q(x) for x in full)


@dataclass
class Field:
    key: str
    label: str
    widget: Any
    var: Any


class ParamToolApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Eval 参数工具 (内部版)")
        root.geometry("1100x720")

        self._runner_thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._log_q: "queue.Queue[str]" = queue.Queue()

        self._profiles = self._load_profiles()

        self.fields: Dict[str, Field] = {}
        self._build_ui()
        self._refresh_profiles_list()
        self._apply_startup_defaults()
        self._poll_logs()

    # ---------- Profiles ----------
    def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
        obj = _safe_read_json(PROFILES_PATH)
        profiles = obj.get("profiles") if isinstance(obj, dict) else None
        if isinstance(profiles, dict):
            return {str(k): v for k, v in profiles.items() if isinstance(v, dict)}
        return {}

    def _save_profiles(self) -> None:
        _safe_write_json(PROFILES_PATH, {"profiles": self._profiles})

    def _refresh_profiles_list(self) -> None:
        names = sorted(self._profiles.keys())
        self.profile_combo["values"] = names
        # IMPORTANT: never auto-select or auto-load profiles on startup,
        # to avoid leaking local paths / keys in the UI by default.

    def _apply_startup_defaults(self) -> None:
        """
        Startup behavior:
        - Always start from a clean/default config (no local paths, no keys)
        - Never auto-load saved profiles
        """
        self.profile_var.set("")
        self._set_cfg(self._default_cfg())

    def _default_cfg(self) -> Dict[str, Any]:
        return {
            "cot": "on",
            "vpn": "off",
            "proxy": "http://127.0.0.1:7897",
            "concurrency": 1,
            "max_retries": 3,
            "answer_json_max_attempts": 3,
            "model_provider": "openai",
            "judge_provider": "openai",
        }

    # ---------- UI ----------
    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: params + logs in a vertical split (prevents advanced settings being visually \"covered\")
        left_split = ttk.PanedWindow(top, orient=tk.VERTICAL)
        left_split.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        params_host = ttk.Frame(left_split)
        logs_host = ttk.Frame(left_split)
        left_split.add(params_host, weight=3)
        left_split.add(logs_host, weight=2)

        # Make params area scrollable (so advanced settings are always reachable)
        params_canvas = tk.Canvas(params_host, highlightthickness=0)
        params_vscroll = ttk.Scrollbar(params_host, orient=tk.VERTICAL, command=params_canvas.yview)
        params_canvas.configure(yscrollcommand=params_vscroll.set)
        params_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        params_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        params_inner = ttk.Frame(params_canvas)
        params_window_id = params_canvas.create_window((0, 0), window=params_inner, anchor="nw")

        def _on_params_configure(_: Any) -> None:
            params_canvas.configure(scrollregion=params_canvas.bbox("all"))

        def _on_canvas_configure(event: Any) -> None:
            # keep inner frame width synced with canvas width
            try:
                params_canvas.itemconfigure(params_window_id, width=event.width)
            except Exception:
                pass

        params_inner.bind("<Configure>", _on_params_configure)
        params_canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scrolling (best-effort). Keeps UX sane when advanced expands.
        def _on_mousewheel(event: Any) -> None:
            try:
                delta = int(-1 * (event.delta / 120))
            except Exception:
                delta = 0
            if delta:
                params_canvas.yview_scroll(delta, "units")

        params_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Right: profiles + actions
        right = ttk.Frame(top)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_profiles_panel(right)
        self._build_params_panel(params_inner)
        self._build_logs_panel(logs_host)

    def _build_profiles_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="配置方案", padding=10)
        box.pack(side=tk.TOP, fill=tk.X)

        self.profile_var = tk.StringVar(value="")
        self.profile_combo = ttk.Combobox(box, textvariable=self.profile_var, state="readonly", width=26)
        self.profile_combo.pack(side=tk.TOP, fill=tk.X)

        btns = ttk.Frame(box)
        btns.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        ttk.Button(btns, text="载入", command=self.on_profile_load).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(btns, text="保存为...", command=self.on_profile_save_as).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=6)
        ttk.Button(btns, text="删除", command=self.on_profile_delete).pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X, pady=10)

        actions = ttk.LabelFrame(parent, text="操作", padding=10)
        actions.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(actions, text="生成命令行", command=self.on_generate_cmd).pack(side=tk.TOP, fill=tk.X)
        ttk.Button(actions, text="从命令行导入设置", command=self.on_import_from_cmd).pack(side=tk.TOP, fill=tk.X, pady=6)
        ttk.Button(actions, text="运行", command=self.on_run).pack(side=tk.TOP, fill=tk.X)
        ttk.Button(actions, text="停止", command=self.on_stop).pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        self.cmd_preview = tk.Text(parent, height=7, width=42, wrap=tk.WORD)
        self.cmd_preview.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.cmd_preview.insert("1.0", "点击【生成命令行】查看将运行的命令。")
        self.cmd_preview.configure(state=tk.DISABLED)

    def _add_field(self, key: str, label: str, row: int, parent: ttk.Frame, *, kind: str = "entry", values: Optional[List[str]] = None, width: int = 46) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        if kind == "combo":
            var = tk.StringVar(value=(values[0] if values else ""))
            w = ttk.Combobox(parent, textvariable=var, state="readonly", values=(values or []), width=width)
        else:
            var = tk.StringVar(value="")
            w = ttk.Entry(parent, textvariable=var, width=width)
        w.grid(row=row, column=1, sticky=tk.W, pady=3)
        self.fields[key] = Field(key=key, label=label, widget=w, var=var)

    def _add_secret_field(self, key: str, label: str, row: int, parent: ttk.Frame, show_var: tk.BooleanVar, *, width: int = 46) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        var = tk.StringVar(value="")
        w = ttk.Entry(parent, textvariable=var, width=width, show="•")
        w.grid(row=row, column=1, sticky=tk.W, pady=3)
        self.fields[key] = Field(key=key, label=label, widget=w, var=var)

        def _sync_show(*_: Any) -> None:
            try:
                w.configure(show="" if show_var.get() else "•")
            except Exception:
                pass

        show_var.trace_add("write", _sync_show)
        ttk.Checkbutton(parent, text="显示", variable=show_var, command=_sync_show).grid(row=row, column=2, sticky=tk.W, padx=(8, 0))

    def _build_params_panel(self, parent: ttk.Frame) -> None:
        # Main params
        main = ttk.LabelFrame(parent, text="主要参数", padding=10)
        main.pack(side=tk.TOP, fill=tk.X)

        r = 0
        self._add_field("input", "--input (数据集xlsx)", r, main); r += 1
        self._add_field("images_root", "--images-root (图片根目录)", r, main); r += 1
        self._add_field("out_dir", "--out-dir (输出目录)", r, main); r += 1

        # File pickers
        pick = ttk.Frame(main)
        pick.grid(row=0, column=2, rowspan=3, padx=(12, 0), sticky=tk.N)
        ttk.Button(pick, text="选择xlsx...", command=self.on_pick_input).pack(fill=tk.X)
        ttk.Button(pick, text="选择图片目录...", command=self.on_pick_images_root).pack(fill=tk.X, pady=6)
        ttk.Button(pick, text="选择输出目录...", command=self.on_pick_out_dir).pack(fill=tk.X)

        self._add_field("cot", "--cot", r, main, kind="combo", values=["on", "off"], width=12); r += 1
        self._add_field("concurrency", "--concurrency", r, main, width=12); r += 1
        self._add_field("max_retries", "--max-retries", r, main, width=12); r += 1

        self._add_field("vpn", "--vpn", r, main, kind="combo", values=["off", "on"], width=12); r += 1
        self._add_field("proxy", "--proxy", r, main); r += 1

        # Model
        model = ttk.LabelFrame(parent, text="被测模型 (Model)", padding=10)
        model.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        r = 0
        self._add_field("model_provider", "--model-provider", r, model, kind="combo", values=["openai", "gemini", "claude"], width=12); r += 1
        self._add_field("model_base_url", "--model-base-url", r, model); r += 1
        self.show_model_key = tk.BooleanVar(value=False)
        self._add_secret_field("model_api_key", "--model-api-key", r, model, self.show_model_key); r += 1
        self._add_field("model_name", "--model-name", r, model); r += 1

        # Judge
        judge = ttk.LabelFrame(parent, text="裁判模型 (Judge)", padding=10)
        judge.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        r = 0
        ttk.Button(judge, text="一键复制被测模型设置", command=self.on_copy_model_to_judge).grid(row=r, column=0, sticky=tk.W, pady=3)
        r += 1
        self._add_field("judge_provider", "--judge-provider", r, judge, kind="combo", values=["openai", "gemini", "claude"], width=12); r += 1
        self._add_field("judge_base_url", "--judge-base-url", r, judge); r += 1
        self.show_judge_key = tk.BooleanVar(value=False)
        self._add_secret_field("judge_api_key", "--judge-api-key", r, judge, self.show_judge_key); r += 1
        self._add_field("judge_name", "--judge-name", r, judge); r += 1

        # Advanced (collapsible)
        self.adv_open = tk.BooleanVar(value=False)
        adv_hdr = ttk.Frame(parent)
        adv_hdr.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(adv_hdr, text="显示详细设置", variable=self.adv_open, command=self._toggle_advanced).pack(side=tk.LEFT)

        self.adv_box = ttk.LabelFrame(parent, text="详细设置", padding=10)
        # not packed by default

        self._build_advanced_panel(self.adv_box)

    def _build_advanced_panel(self, parent: ttk.LabelFrame) -> None:
        # Split into two columns
        left = ttk.Frame(parent)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(parent)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        r = 0
        self._add_field("config", "--config (YAML)", r, left); r += 1
        self._add_field("sheet", "--sheet", r, left); r += 1
        self._add_field("limit", "--limit", r, left, width=12); r += 1
        self._add_field("answer_json_max_attempts", "--answer-json-max-attempts", r, left, width=12); r += 1

        self._add_field("model_concurrency", "--model-concurrency", r, left, width=12); r += 1
        self._add_field("judge_concurrency", "--judge-concurrency", r, left, width=12); r += 1
        self._add_field("retry_base_delay_s", "--retry-base-delay-s", r, left, width=12); r += 1
        self._add_field("retry_max_delay_s", "--retry-max-delay-s", r, left, width=12); r += 1

        r2 = 0
        self._add_field("model_timeout_s", "--model-timeout-s", r2, right, width=12); r2 += 1
        self._add_field("model_temperature", "--model-temperature", r2, right, width=12); r2 += 1
        self._add_field("model_max_tokens", "--model-max-tokens", r2, right, width=12); r2 += 1

        self._add_field("judge_timeout_s", "--judge-timeout-s", r2, right, width=12); r2 += 1
        self._add_field("judge_temperature", "--judge-temperature", r2, right, width=12); r2 += 1
        self._add_field("judge_max_tokens", "--judge-max-tokens", r2, right, width=12); r2 += 1

        pick = ttk.Frame(left)
        pick.grid(row=0, column=2, rowspan=2, padx=(12, 0), sticky=tk.N)
        ttk.Button(pick, text="选择YAML...", command=self.on_pick_config).pack(fill=tk.X)

    def _toggle_advanced(self) -> None:
        if self.adv_open.get():
            self.adv_box.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))
        else:
            self.adv_box.pack_forget()

    def _build_logs_panel(self, parent: ttk.Frame) -> None:
        logs = ttk.LabelFrame(parent, text="运行日志", padding=10)
        logs.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(10, 0))

        self.log_text = tk.Text(logs, height=18, wrap=tk.NONE)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        yscroll = ttk.Scrollbar(logs, orient=tk.VERTICAL, command=self.log_text.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=yscroll.set)

    # ---------- Config get/set ----------
    def _get_cfg(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}

        def get_s(k: str) -> str:
            return (self.fields[k].var.get() if k in self.fields else "").strip()

        # strings
        for k in [
            "config",
            "input",
            "sheet",
            "images_root",
            "out_dir",
            "proxy",
            "model_base_url",
            "model_api_key",
            "model_name",
            "judge_base_url",
            "judge_api_key",
            "judge_name",
        ]:
            v = get_s(k)
            if not _is_blank(v):
                cfg[k] = v

        # enums
        for k in ["cot", "vpn", "model_provider", "judge_provider"]:
            v = get_s(k)
            if not _is_blank(v):
                cfg[k] = v

        # ints
        for k in [
            "concurrency",
            "model_concurrency",
            "judge_concurrency",
            "max_retries",
            "limit",
            "answer_json_max_attempts",
            "model_max_tokens",
            "judge_max_tokens",
        ]:
            v = _int_or_none(get_s(k))
            if v is not None:
                cfg[k] = v

        # floats
        for k in [
            "retry_base_delay_s",
            "retry_max_delay_s",
            "model_timeout_s",
            "model_temperature",
            "judge_timeout_s",
            "judge_temperature",
        ]:
            v = _float_or_none(get_s(k))
            if v is not None:
                cfg[k] = v

        return cfg

    def _set_cfg(self, cfg: Dict[str, Any]) -> None:
        # Defaults first
        merged = self._default_cfg()
        merged.update(cfg or {})

        def set_v(k: str, v: Any) -> None:
            if k in self.fields:
                self.fields[k].var.set("" if v is None else str(v))

        for k, v in merged.items():
            set_v(k, v)

    # ---------- Actions ----------
    def on_pick_input(self) -> None:
        p = filedialog.askopenfilename(title="选择数据集 xlsx", filetypes=[("Excel", "*.xlsx"), ("All", "*")])
        if p:
            self.fields["input"].var.set(p)

    def on_pick_images_root(self) -> None:
        p = filedialog.askdirectory(title="选择 images-root 目录")
        if p:
            self.fields["images_root"].var.set(p)

    def on_pick_out_dir(self) -> None:
        p = filedialog.askdirectory(title="选择 out-dir 目录")
        if p:
            self.fields["out_dir"].var.set(p)

    def on_pick_config(self) -> None:
        p = filedialog.askopenfilename(title="选择 YAML 配置", filetypes=[("YAML", "*.yml *.yaml"), ("All", "*")])
        if p:
            self.fields["config"].var.set(p)

    def on_profile_load(self) -> None:
        name = (self.profile_var.get() or "").strip()
        if not name:
            messagebox.showwarning("提示", "请选择一个配置方案")
            return
        cfg = self._profiles.get(name)
        if not isinstance(cfg, dict):
            messagebox.showwarning("提示", "配置不存在或已损坏")
            return
        self._set_cfg(cfg)

    def on_profile_save_as(self) -> None:
        name = tk.simpledialog.askstring("保存配置", "给这次参数设置起个名字：")
        if not name:
            return
        name = name.strip()
        if not name:
            return

        cfg = self._get_cfg()
        self._profiles[name] = cfg
        self._save_profiles()
        self._refresh_profiles_list()
        self.profile_var.set(name)
        messagebox.showinfo("保存成功", f"已保存配置：{name}\n文件：{PROFILES_PATH}")

    def on_profile_delete(self) -> None:
        name = (self.profile_var.get() or "").strip()
        if not name:
            messagebox.showwarning("提示", "请选择一个配置方案")
            return
        if name not in self._profiles:
            messagebox.showwarning("提示", "配置不存在")
            return
        if not messagebox.askyesno("确认删除", f"确定删除配置：{name} ？"):
            return
        self._profiles.pop(name, None)
        self._save_profiles()
        self._refresh_profiles_list()
        self._apply_startup_defaults()

    def _validate_before_run(self, cfg: Dict[str, Any]) -> Tuple[bool, str]:
        if not os.path.exists(EVAL_SCRIPT):
            return False, f"找不到脚本：{EVAL_SCRIPT}"
        inp = cfg.get("input")
        if not inp:
            return False, "必须填写 --input"
        # Allow non-existent on Linux? For Windows internal tool, validate.
        if not os.path.exists(str(inp)):
            return False, f"--input 不存在：{inp}"
        return True, ""

    def on_generate_cmd(self) -> None:
        cfg = self._get_cfg()
        argv = _build_arg_list(cfg)
        cmd = _format_cmd_for_display(argv)
        self.cmd_preview.configure(state=tk.NORMAL)
        self.cmd_preview.delete("1.0", tk.END)
        self.cmd_preview.insert("1.0", cmd)
        self.cmd_preview.configure(state=tk.DISABLED)

    def on_copy_model_to_judge(self) -> None:
        """
        Copy model settings into judge settings.
        Includes provider/base_url/api_key/model-name (judge-name) by default.
        """
        for src, dst in [
            ("model_provider", "judge_provider"),
            ("model_base_url", "judge_base_url"),
            ("model_api_key", "judge_api_key"),
            ("model_name", "judge_name"),
        ]:
            if src in self.fields and dst in self.fields:
                self.fields[dst].var.set(self.fields[src].var.get())

    def _parse_cmd_to_cfg(self, cmd: str) -> Dict[str, Any]:
        """
        Parse a command line string and extract known flags into a cfg dict.
        Unknown flags are ignored.
        If formatting is invalid (e.g. missing value), raises ValueError.
        """
        cmd = (cmd or "").strip()
        if not cmd:
            return {}
        try:
            tokens = shlex.split(cmd, posix=(os.name != "nt"))
        except Exception as e:
            raise ValueError(f"命令行解析失败：{type(e).__name__}: {e}") from e

        # Drop leading python executable and/or script path if present
        cleaned: List[str] = []
        for t in tokens:
            # ignore "python"/"python3"/sys.executable
            base = os.path.basename(t).lower()
            if base in {"python", "python.exe", "python3", "python3.exe"}:
                continue
            # ignore eval script path
            if os.path.basename(t) == os.path.basename(EVAL_SCRIPT):
                continue
            cleaned.append(t)

        # Known flags mapping
        value_flags: Dict[str, str] = {
            "--config": "config",
            "--input": "input",
            "--sheet": "sheet",
            "--images-root": "images_root",
            "--out-dir": "out_dir",
            "--concurrency": "concurrency",
            "--model-concurrency": "model_concurrency",
            "--judge-concurrency": "judge_concurrency",
            "--max-retries": "max_retries",
            "--retry-base-delay-s": "retry_base_delay_s",
            "--retry-max-delay-s": "retry_max_delay_s",
            "--limit": "limit",
            "--cot": "cot",
            "--answer-json-max-attempts": "answer_json_max_attempts",
            "--vpn": "vpn",
            "--proxy": "proxy",
            "--model-provider": "model_provider",
            "--model-base-url": "model_base_url",
            "--model-api-key": "model_api_key",
            "--model-name": "model_name",
            "--model-timeout-s": "model_timeout_s",
            "--model-temperature": "model_temperature",
            "--model-max-tokens": "model_max_tokens",
            "--judge-provider": "judge_provider",
            "--judge-base-url": "judge_base_url",
            "--judge-api-key": "judge_api_key",
            "--judge-name": "judge_name",
            "--judge-timeout-s": "judge_timeout_s",
            "--judge-temperature": "judge_temperature",
            "--judge-max-tokens": "judge_max_tokens",
        }
        ignored_bool_flags = {"--judge-enable"}  # deprecated/no-op

        int_keys = {
            "concurrency",
            "model_concurrency",
            "judge_concurrency",
            "max_retries",
            "limit",
            "answer_json_max_attempts",
            "model_max_tokens",
            "judge_max_tokens",
        }
        float_keys = {
            "retry_base_delay_s",
            "retry_max_delay_s",
            "model_timeout_s",
            "model_temperature",
            "judge_timeout_s",
            "judge_temperature",
        }

        out: Dict[str, Any] = {}
        i = 0
        while i < len(cleaned):
            tok = cleaned[i]
            if tok in ignored_bool_flags:
                i += 1
                continue

            if tok.startswith("--") and "=" in tok:
                k, v = tok.split("=", 1)
                if k in value_flags:
                    out[value_flags[k]] = v
                i += 1
                continue

            if tok in value_flags:
                if i + 1 >= len(cleaned) or cleaned[i + 1].startswith("--"):
                    raise ValueError(f"参数 {tok} 缺少值")
                out[value_flags[tok]] = cleaned[i + 1]
                i += 2
                continue

            # Unknown token: ignore
            i += 1

        # Type conversions
        for k in list(out.keys()):
            if k in int_keys:
                v = _int_or_none(str(out[k]))
                if v is None:
                    raise ValueError(f"参数 {k} 需要整数，但得到：{out[k]}")
                out[k] = v
            elif k in float_keys:
                v = _float_or_none(str(out[k]))
                if v is None:
                    raise ValueError(f"参数 {k} 需要数字，但得到：{out[k]}")
                out[k] = v

        return out

    def on_import_from_cmd(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("从命令行导入设置")
        dlg.geometry("820x320")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text="粘贴一整行命令（会尽可能解析并导入已支持的参数；未知参数会忽略）：").pack(
            side=tk.TOP, anchor=tk.W, padx=10, pady=(10, 6)
        )
        txt = tk.Text(dlg, height=10, wrap=tk.WORD)
        txt.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10)

        btns = ttk.Frame(dlg)
        btns.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        def do_import() -> None:
            raw = txt.get("1.0", tk.END).strip()
            try:
                parsed = self._parse_cmd_to_cfg(raw)
            except Exception as e:
                messagebox.showerror("导入失败", str(e), parent=dlg)
                return
            # Start from defaults to avoid carrying over hidden/old sensitive values
            merged = self._default_cfg()
            merged.update(parsed)
            self._set_cfg(merged)
            dlg.destroy()

        ttk.Button(btns, text="导入", command=do_import).pack(side=tk.RIGHT)
        ttk.Button(btns, text="取消", command=dlg.destroy).pack(side=tk.RIGHT, padx=(0, 8))

    def on_run(self) -> None:
        if self._proc is not None:
            messagebox.showwarning("正在运行", "已有任务在运行中，请先停止或等待完成")
            return

        cfg = self._get_cfg()
        ok, msg = self._validate_before_run(cfg)
        if not ok:
            messagebox.showerror("参数错误", msg)
            return

        argv = _build_arg_list(cfg)
        # Run with current interpreter
        full = [sys.executable] + argv

        self._append_log("\n=== RUN ===\n" + _format_cmd_for_display(argv) + "\n")

        def runner() -> None:
            try:
                # Force UTF-8 for stdout decoding on Windows to avoid GBK UnicodeDecodeError.
                # - encoding/errors affect how *this* process decodes stdout.
                # - PYTHONIOENCODING/PYTHONUTF8 influence how the child process encodes stdout.
                child_env = os.environ.copy()
                child_env.setdefault("PYTHONIOENCODING", "utf-8")
                child_env.setdefault("PYTHONUTF8", "1")
                self._proc = subprocess.Popen(
                    full,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(EVAL_SCRIPT),
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=child_env,
                )
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    self._log_q.put(line)
                rc = self._proc.wait()
                self._log_q.put(f"\n=== DONE (exit={rc}) ===\n")
            except Exception as e:
                self._log_q.put(f"\n=== ERROR: {type(e).__name__}: {e} ===\n")
            finally:
                self._proc = None

        self._runner_thread = threading.Thread(target=runner, daemon=True)
        self._runner_thread.start()

    def on_stop(self) -> None:
        p = self._proc
        if p is None:
            return
        try:
            p.terminate()
            self._append_log("\n=== STOP requested ===\n")
        except Exception as e:
            self._append_log(f"\n=== STOP failed: {type(e).__name__}: {e} ===\n")

    # ---------- Logging ----------
    def _append_log(self, s: str) -> None:
        self.log_text.insert(tk.END, s)
        self.log_text.see(tk.END)

    def _poll_logs(self) -> None:
        try:
            while True:
                s = self._log_q.get_nowait()
                self._append_log(s)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_logs)


def main() -> None:
    # Ensure tkinter dialog helper available
    try:
        import tkinter.simpledialog  # noqa: F401
    except Exception:
        pass

    root = tk.Tk()
    # Use themed widgets
    try:
        style = ttk.Style()
        if os.name == "nt":
            style.theme_use("vista")
        else:
            style.theme_use("clam")
    except Exception:
        pass

    ParamToolApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
