import json
import os
import queue
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
    add_bool("--judge-enable", bool(cfg.get("judge_enable")))
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
        self._apply_default_profile_if_any()
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
        if names and self.profile_var.get() not in names:
            self.profile_var.set(names[0])

    def _apply_default_profile_if_any(self) -> None:
        if not self._profiles:
            # Pre-fill with a reasonable template
            self._set_cfg(self._default_cfg())
            return
        name = self.profile_var.get()
        if name and name in self._profiles:
            self._set_cfg(self._profiles[name])

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

        # Left: params
        left = ttk.Frame(top)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right: profiles + actions
        right = ttk.Frame(top)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_profiles_panel(right)
        self._build_params_panel(left)
        self._build_logs_panel(left)

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
        ttk.Button(actions, text="复制命令行", command=self.on_copy_cmd).pack(side=tk.TOP, fill=tk.X, pady=6)
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
        self._add_field("model_api_key", "--model-api-key", r, model); r += 1
        self._add_field("model_name", "--model-name", r, model); r += 1

        # Judge
        judge = ttk.LabelFrame(parent, text="裁判模型 (Judge)", padding=10)
        judge.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        r = 0
        self.judge_enable_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(judge, text="--judge-enable (已废弃, no-op)", variable=self.judge_enable_var).grid(row=r, column=0, sticky=tk.W, pady=3)
        r += 1
        self._add_field("judge_provider", "--judge-provider", r, judge, kind="combo", values=["openai", "gemini", "claude"], width=12); r += 1
        self._add_field("judge_base_url", "--judge-base-url", r, judge); r += 1
        self._add_field("judge_api_key", "--judge-api-key", r, judge); r += 1
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

        cfg["judge_enable"] = bool(self.judge_enable_var.get())

        return cfg

    def _set_cfg(self, cfg: Dict[str, Any]) -> None:
        # Defaults first
        merged = self._default_cfg()
        merged.update(cfg or {})

        def set_v(k: str, v: Any) -> None:
            if k in self.fields:
                self.fields[k].var.set("" if v is None else str(v))

        for k, v in merged.items():
            if k == "judge_enable":
                self.judge_enable_var.set(bool(v))
            else:
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
        self._apply_default_profile_if_any()

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

    def on_copy_cmd(self) -> None:
        cfg = self._get_cfg()
        argv = _build_arg_list(cfg)
        cmd = _format_cmd_for_display(argv)
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(cmd)
            self.root.update()
            messagebox.showinfo("已复制", "命令行已复制到剪贴板")
        except Exception as e:
            messagebox.showerror("复制失败", str(e))

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
                self._proc = subprocess.Popen(
                    full,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(EVAL_SCRIPT),
                    text=True,
                    bufsize=1,
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
