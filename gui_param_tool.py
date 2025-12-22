import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import tkinter as tk
import contextlib
import io
import signal
import time
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd


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
    add("--root", cfg.get("root"))
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
    add("--majority-vote", cfg.get("majority_vote"))
    add("--mcq-cardinality-hint", cfg.get("mcq_cardinality_hint"))

    # Network
    add("--vpn", cfg.get("vpn"))
    add("--proxy", cfg.get("proxy"))

    # Model
    add("--api-protocol", cfg.get("model_provider"))
    add("--model-base-url", cfg.get("model_base_url"))
    add("--model-api-key", cfg.get("model_api_key"))
    add("--model-name", cfg.get("model_name"))
    add("--model-timeout-s", cfg.get("model_timeout_s"))
    add("--model-temperature", cfg.get("model_temperature"))
    add("--model-top-p", cfg.get("model_top_p"))
    add("--model-max-tokens", cfg.get("model_max_tokens"))

    # Judge
    add("--judge-api-protocol", cfg.get("judge_provider"))
    add("--judge-base-url", cfg.get("judge_base_url"))
    add("--judge-api-key", cfg.get("judge_api_key"))
    add("--judge-name", cfg.get("judge_name"))
    add("--judge-timeout-s", cfg.get("judge_timeout_s"))
    add("--judge-temperature", cfg.get("judge_temperature"))
    add("--judge-top-p", cfg.get("judge_top_p"))
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
        root.title("评测小程序")
        root.geometry("1100x720")

        self._runner_thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._cancel_event: Optional[threading.Event] = None
        self._log_q: "queue.Queue[str]" = queue.Queue()
        self._last_cfg: Dict[str, Any] = {}

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
        base_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(__file__)
        return {
            "cot": "off",
            "mcq_cardinality_hint": "on",
            "vpn": "off",
            "proxy": "http://127.0.0.1:7897",
            "concurrency": 1,
            "max_retries": 3,
            "answer_json_max_attempts": 3,
            "majority_vote": 1,
            "out_dir": os.path.join(base_dir, "out_run"),
            "model_provider": "OpenAI",
            "judge_provider": "OpenAI",
            "model_temperature": 0,
            "judge_temperature": 0,
            "model_top_p": 0.75,
            "judge_top_p": 0.75,
            "model_max_tokens": 10000,
            "judge_max_tokens": 10000,
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
        ttk.Button(actions, text="连通性测试", command=self.on_connectivity_test).pack(side=tk.TOP, fill=tk.X)
        ttk.Button(actions, text="桑基图绘制", command=self.on_sankey).pack(side=tk.TOP, fill=tk.X, pady=(6, 0))
        ttk.Button(actions, text="数据集统计（饼图）", command=self.on_pie).pack(side=tk.TOP, fill=tk.X)
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
        self._add_field("root", "--root (数据集根目录，需包含 dataset.xlsx 与 images/)", r, main); r += 1
        self._add_field("majority_vote", "--majority-vote (同题多次调用答题模型并多数投票，>=1)", r, main, width=12); r += 1

        # File pickers
        pick = ttk.Frame(main)
        pick.grid(row=0, column=2, rowspan=2, padx=(12, 0), sticky=tk.N)
        ttk.Button(pick, text="选择数据集根目录...", command=self.on_pick_root).pack(fill=tk.X)

        self._add_field("cot", "--cot (思维链开关：on=输出cot思维链，off=不输出思维链)", r, main, kind="combo", values=["on", "off"], width=12); r += 1
        self._add_field(
            "mcq_cardinality_hint",
            "--mcq-cardinality-hint (选择题提示：告知单选/多选 on/off)",
            r,
            main,
            kind="combo",
            values=["on", "off"],
            width=12,
        ); r += 1
        self._add_field("concurrency", "--concurrency (并发数：同时向大模型并发多少条消息)", r, main, width=12); r += 1
        self._add_field("max_retries", "--max-retries (网络/接口失败重试次数，指数退避)", r, main, width=12); r += 1

        self._add_field("vpn", "--vpn (如需VPN/代理访问受限API请开启)", r, main, kind="combo", values=["off", "on"], width=12); r += 1
        self._add_field("proxy", "--proxy (代理地址，如 http://127.0.0.1:7897)", r, main); r += 1

        # Model
        model = ttk.LabelFrame(parent, text="被测模型 (Model)", padding=10)
        model.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        r = 0
        self._add_field(
            "model_provider",
            "--api-protocol (API协议：OpenAI/Anthropic/Google)",
            r,
            model,
            kind="combo",
            values=["OpenAI", "Anthropic", "Google"],
            width=12,
        ); r += 1
        self._add_field("model_base_url", "--model-base-url (接口地址，如 https://api.openai.com)", r, model); r += 1
        self.show_model_key = tk.BooleanVar(value=False)
        self._add_secret_field("model_api_key", "--model-api-key (答题模型密钥)", r, model, self.show_model_key); r += 1
        self._add_field("model_name", "--model-name (答题模型名称)", r, model); r += 1

        # Judge
        judge = ttk.LabelFrame(parent, text="裁判模型 (Judge)", padding=10)
        judge.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        r = 0
        ttk.Button(judge, text="一键复制被测模型设置", command=self.on_copy_model_to_judge).grid(row=r, column=0, sticky=tk.W, pady=3)
        r += 1
        self._add_field(
            "judge_provider",
            "--judge-api-protocol (API协议：OpenAI/Anthropic/Google)",
            r,
            judge,
            kind="combo",
            values=["OpenAI", "Anthropic", "Google"],
            width=12,
        ); r += 1
        self._add_field("judge_base_url", "--judge-base-url (接口地址)", r, judge); r += 1
        self.show_judge_key = tk.BooleanVar(value=False)
        self._add_secret_field("judge_api_key", "--judge-api-key (裁判模型密钥)", r, judge, self.show_judge_key); r += 1
        self._add_field("judge_name", "--judge-name (裁判模型名称)", r, judge); r += 1

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
        self._add_field("config", "--config (YAML配置文件路径，优先级低于命令行显式参数)", r, left); r += 1
        self._add_field("out_dir", "--out-dir (结果输出目录，默认 out_run/)", r, left); r += 1
        self._add_field("sheet", "--sheet (Excel工作表名，留空=第一个sheet)", r, left); r += 1
        self._add_field("limit", "--limit (只跑前N题)", r, left, width=12); r += 1
        self._add_field("answer_json_max_attempts", "--answer-json-max-attempts (答题输出JSON解析失败时的重问次数)", r, left, width=12); r += 1

        self._add_field("model_concurrency", "--model-concurrency (答题模型最大并发请求数)", r, left, width=12); r += 1
        self._add_field("judge_concurrency", "--judge-concurrency (裁判模型最大并发请求数)", r, left, width=12); r += 1
        self._add_field("retry_base_delay_s", "--retry-base-delay-s (重试退避基准秒数)", r, left, width=12); r += 1
        self._add_field("retry_max_delay_s", "--retry-max-delay-s (重试退避最大秒数)", r, left, width=12); r += 1

        r2 = 0
        self._add_field("model_timeout_s", "--model-timeout-s (答题接口超时秒数)", r2, right, width=12); r2 += 1
        self._add_field("model_temperature", "--model-temperature (答题采样温度，默认0)", r2, right, width=12); r2 += 1
        self._add_field("model_top_p", "--model-top-p (答题top-p，默认0.75)", r2, right, width=12); r2 += 1
        self._add_field("model_max_tokens", "--model-max-tokens (答题最大输出token)", r2, right, width=12); r2 += 1

        self._add_field("judge_timeout_s", "--judge-timeout-s (裁判接口超时秒数)", r2, right, width=12); r2 += 1
        self._add_field("judge_temperature", "--judge-temperature (裁判采样温度，默认0)", r2, right, width=12); r2 += 1
        self._add_field("judge_top_p", "--judge-top-p (裁判top-p，默认0.75)", r2, right, width=12); r2 += 1
        self._add_field("judge_max_tokens", "--judge-max-tokens (裁判最大输出token)", r2, right, width=12); r2 += 1

        pick = ttk.Frame(left)
        pick.grid(row=0, column=2, rowspan=3, padx=(12, 0), sticky=tk.N)
        ttk.Button(pick, text="选择YAML...", command=self.on_pick_config).pack(fill=tk.X)
        ttk.Button(pick, text="选择输出目录...", command=self.on_pick_out_dir).pack(fill=tk.X, pady=6)

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
            "root",
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
        for k in ["cot", "mcq_cardinality_hint", "vpn", "model_provider", "judge_provider"]:
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
            "majority_vote",
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
            "model_top_p",
            "judge_timeout_s",
            "judge_temperature",
            "judge_top_p",
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
    def on_pick_root(self) -> None:
        p = filedialog.askdirectory(title="选择数据集根目录（包含 dataset.xlsx 与 images/）")
        if p:
            self.fields["root"].var.set(p)

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
        root = (cfg.get("root") or "").strip()
        inp = (cfg.get("input") or "").strip()
        if not root and not inp:
            return False, "必须填写 --root 或 --input"
        if inp:
            # Allow non-existent on Linux? For Windows internal tool, validate.
            if not os.path.exists(str(inp)):
                return False, f"--input 不存在：{inp}"
        else:
            dataset = os.path.join(root, "dataset.xlsx")
            if not os.path.exists(dataset):
                return False, f"--root 下找不到 dataset.xlsx：{dataset}"
        mv = cfg.get("majority_vote")
        if mv is not None:
            try:
                mv_i = int(mv)
                if mv_i < 1:
                    return False, "--majority-vote 必须为 >= 1"
            except Exception:
                return False, "--majority-vote 必须为整数"
        return True, ""

    def on_generate_cmd(self) -> None:
        cfg = self._get_cfg()
        self._last_cfg = cfg
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
            "--root": "root",
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
            "--mcq-cardinality-hint": "mcq_cardinality_hint",
            "--answer-json-max-attempts": "answer_json_max_attempts",
            "--majority-vote": "majority_vote",
            "--vpn": "vpn",
            "--proxy": "proxy",
            "--model-provider": "model_provider",
            "--api-protocol": "model_provider",
            "--model-base-url": "model_base_url",
            "--model-api-key": "model_api_key",
            "--model-name": "model_name",
            "--model-timeout-s": "model_timeout_s",
            "--model-temperature": "model_temperature",
            "--model-top-p": "model_top_p",
            "--model-max-tokens": "model_max_tokens",
            "--judge-provider": "judge_provider",
            "--judge-api-protocol": "judge_provider",
            "--judge-base-url": "judge_base_url",
            "--judge-api-key": "judge_api_key",
            "--judge-name": "judge_name",
            "--judge-timeout-s": "judge_timeout_s",
            "--judge-temperature": "judge_temperature",
            "--judge-top-p": "judge_top_p",
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
            "majority_vote",
            "model_max_tokens",
            "judge_max_tokens",
        }
        float_keys = {
            "retry_base_delay_s",
            "retry_max_delay_s",
            "model_timeout_s",
            "model_temperature",
            "model_top_p",
            "judge_timeout_s",
            "judge_temperature",
            "judge_top_p",
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
        if self._is_running():
            messagebox.showwarning("正在运行", "已有任务在运行中，请先停止或等待完成")
            return

        cfg = self._get_cfg()
        self._last_cfg = cfg
        ok, msg = self._validate_before_run(cfg)
        if not ok:
            messagebox.showerror("参数错误", msg)
            return

        argv = _build_arg_list(cfg)
        # In frozen (packaged) mode, run eval in-process; otherwise run as a subprocess.
        is_frozen = bool(getattr(sys, "frozen", False))
        full = [sys.executable] + argv

        self._append_log("\n=== RUN ===\n" + _format_cmd_for_display(argv) + "\n")

        def runner() -> None:
            try:
                if is_frozen:
                    # Run eval_questions in-process (PyInstaller executable cannot spawn python.exe reliably).
                    import eval_questions  # type: ignore

                    class _QWriter(io.TextIOBase):
                        def __init__(self, q: "queue.Queue[str]"):
                            self.q = q

                        def write(self, s: str) -> int:
                            if s:
                                self.q.put(s)
                            return len(s or "")

                        def flush(self) -> None:
                            return

                    self._cancel_event = threading.Event()
                    out = _QWriter(self._log_q)
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                        try:
                            # argv starts with eval_questions.py; pass flags only.
                            eval_questions.cli_main(argv[1:], cancel_event=self._cancel_event)
                            self._log_q.put("\n=== DONE (exit=0) ===\n")
                        except SystemExit as e:
                            code = int(getattr(e, "code", 1) or 0)
                            self._log_q.put(f"\n=== DONE (exit={code}) ===\n")
                        finally:
                            self._cancel_event = None
                            self._flush_plot_buffer()
                else:
                    # Force UTF-8 for stdout decoding on Windows to avoid GBK UnicodeDecodeError.
                    child_env = os.environ.copy()
                    child_env.setdefault("PYTHONIOENCODING", "utf-8")
                    child_env.setdefault("PYTHONUTF8", "1")
                    creationflags = 0
                    popen_kwargs: Dict[str, Any] = {}
                    if os.name == "nt":
                        # Best-effort: isolate process group so CTRL/CLOSE can target it (may vary by host).
                        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        if creationflags:
                            popen_kwargs["creationflags"] = creationflags
                    else:
                        # Create a new session so we can terminate the whole process group.
                        popen_kwargs["start_new_session"] = True
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
                        **popen_kwargs,
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
        # Subprocess mode (preferred in dev): terminate whole process group, then force-kill if needed.
        p = self._proc
        if p is not None:
            self._append_log("\n=== STOP requested ===\n")

            def killer() -> None:
                try:
                    if os.name != "nt":
                        try:
                            os.killpg(p.pid, signal.SIGTERM)
                        except Exception:
                            p.terminate()
                    else:
                        p.terminate()

                    deadline = time.time() + 3.0
                    while time.time() < deadline:
                        if p.poll() is not None:
                            return
                        time.sleep(0.1)

                    # Still alive: hard kill
                    if os.name != "nt":
                        try:
                            os.killpg(p.pid, signal.SIGKILL)
                        except Exception:
                            p.kill()
                    else:
                        p.kill()
                except Exception as e:
                    self._log_q.put(f"\n=== STOP failed: {type(e).__name__}: {e} ===\n")

            threading.Thread(target=killer, daemon=True).start()
            return

        # In-process (frozen) mode: cooperative cancellation.
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._append_log("\n=== STOP requested (cancel) ===\n")

    def _is_running(self) -> bool:
        if self._proc is not None:
            return True
        t = self._runner_thread
        return bool(t is not None and t.is_alive())

    def on_connectivity_test(self) -> None:
        """
        Best-effort connectivity test for model/judge settings.
        Runs in a background thread and logs results to the GUI.
        """
        if self._runner_thread is not None and self._runner_thread.is_alive():
            messagebox.showwarning("正在运行", "有任务在运行中，请等待完成或先停止")
            return

        cfg = self._get_cfg()

        def runner() -> None:
            try:
                self._log_q.put("\n=== CONNECTIVITY TEST ===\n")
                import asyncio
                import eval_questions  # type: ignore

                # Build minimal run config for http client settings
                run_cfg = eval_questions.RunConfig(
                    input_path="",
                    sheet_name=None,
                    images_root=cfg.get("images_root", "") or "",
                    out_dir=cfg.get("out_dir", "") or "",
                    concurrency=1,
                    model_concurrency=1,
                    judge_concurrency=1,
                    max_retries=0,
                    retry_base_delay_s=0.1,
                    retry_max_delay_s=0.1,
                    skip_image_missing=True,
                    limit=None,
                    cot=str(cfg.get("cot", "on") or "on"),
                    answer_json_max_attempts=int(cfg.get("answer_json_max_attempts", 1) or 1),
                    majority_vote=1,
                    vpn=str(cfg.get("vpn", "off") or "off"),
                    proxy=str(cfg.get("proxy", "") or ""),
                )

                async def _test_one(name: str, pcfg: eval_questions.ProviderConfig) -> None:
                    prov = eval_questions.LLMProvider(pcfg, run_cfg)
                    t0 = eval_questions.now_ms()
                    resp = await prov.call("Reply with ONLY a JSON object: {\"ok\": true}", image_data_url=None)
                    t1 = eval_questions.now_ms()
                    err = resp.get("error")
                    st = resp.get("status")
                    if err or (isinstance(st, int) and st >= 400):
                        self._log_q.put(f"{name}: FAIL  status={st}  error={err}\n")
                    else:
                        self._log_q.put(f"{name}: OK  status={st}  ms={t1 - t0}\n")

                # Use small max_tokens to minimize cost
                model_pcfg = eval_questions.ProviderConfig(
                    provider=eval_questions.normalize_api_protocol(str(cfg.get("model_provider", "OpenAI") or "OpenAI")),
                    base_url=str(cfg.get("model_base_url", "") or ""),
                    api_key=str(cfg.get("model_api_key", "") or ""),
                    model=str(cfg.get("model_name", "") or ""),
                    timeout_s=float(cfg.get("model_timeout_s", 20) or 20),
                    temperature=0.0,
                    max_tokens=16,
                )
                judge_pcfg = eval_questions.ProviderConfig(
                    provider=eval_questions.normalize_api_protocol(str(cfg.get("judge_provider", model_pcfg.provider) or model_pcfg.provider)),
                    base_url=str(cfg.get("judge_base_url", model_pcfg.base_url) or model_pcfg.base_url),
                    api_key=str(cfg.get("judge_api_key", model_pcfg.api_key) or model_pcfg.api_key),
                    model=str(cfg.get("judge_name", model_pcfg.model) or model_pcfg.model),
                    timeout_s=float(cfg.get("judge_timeout_s", 20) or 20),
                    temperature=0.0,
                    max_tokens=16,
                )

                asyncio.run(_test_one("Model", model_pcfg))
                asyncio.run(_test_one("Judge", judge_pcfg))
                self._log_q.put("=== CONNECTIVITY TEST DONE ===\n")
            except Exception as e:
                self._log_q.put(f"\n=== CONNECTIVITY TEST ERROR: {type(e).__name__}: {e} ===\n")

        self._runner_thread = threading.Thread(target=runner, daemon=True)
        self._runner_thread.start()

    def on_sankey(self) -> None:
        fields_all = [
            "Question_Type",
            "Subfield",
            "Image_Dependency",
            "Academic_Level",
            "Difficulty",
            "Third-level field",
            "model_correct",
        ]

        dlg = tk.Toplevel(self.root)
        dlg.title("桑基图绘制")
        dlg.geometry("760x420")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text="选择字段并调整顺序（至少2个）。可多次绘制，输出到 out-dir。").pack(anchor=tk.W, padx=10, pady=(10, 6))

        body = ttk.Frame(dlg, padding=10)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(body)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="可选字段").pack(anchor=tk.W)
        lb_avail = tk.Listbox(left, selectmode=tk.EXTENDED, height=12)
        for f in fields_all:
            lb_avail.insert(tk.END, f)
        lb_avail.pack(fill=tk.BOTH, expand=True)

        mid = ttk.Frame(body)
        mid.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(mid, text="添加 →", command=lambda: self._move_listbox_items(lb_avail, lb_sel)).pack(fill=tk.X, pady=(22, 6))
        ttk.Button(mid, text="← 移除", command=lambda: self._remove_selected_items(lb_sel)).pack(fill=tk.X)

        ttk.Label(right, text="已选字段（顺序即桑基图层级）").pack(anchor=tk.W)
        lb_sel = tk.Listbox(right, selectmode=tk.EXTENDED, height=12)
        lb_sel.pack(fill=tk.BOTH, expand=True)

        order_btns = ttk.Frame(right)
        order_btns.pack(fill=tk.X, pady=6)
        ttk.Button(order_btns, text="上移", command=lambda: self._shift_selected(lb_sel, -1)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(order_btns, text="下移", command=lambda: self._shift_selected(lb_sel, +1)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=6)

        btns = ttk.Frame(dlg, padding=10)
        btns.pack(fill=tk.X)

        def do_draw() -> None:
            fields = [lb_sel.get(i) for i in range(lb_sel.size())]
            if len(fields) < 2:
                messagebox.showwarning("提示", "请至少选择2个字段", parent=dlg)
                return
            needs_mc = ("model_correct" in fields)
            try:
                evaluated_xlsx = None
                if needs_mc:
                    evaluated_xlsx = filedialog.askopenfilename(
                        title="选择模型评测结果 xlsx（包含 model_correct 列）",
                        filetypes=[("Excel", "*.xlsx"), ("All", "*")],
                    )
                    if not evaluated_xlsx:
                        return
                out_path = self._draw_sankey(fields, evaluated_xlsx_path=evaluated_xlsx)
                self._notify_info("绘制成功", f"桑基图已输出：\n{out_path}")
                dlg.destroy()
            except Exception as e:
                messagebox.showerror("绘制失败", f"{type(e).__name__}: {e}", parent=dlg)

        ttk.Button(btns, text="绘制", command=do_draw).pack(side=tk.RIGHT)
        ttk.Button(btns, text="取消", command=dlg.destroy).pack(side=tk.RIGHT, padx=(0, 8))

    def on_pie(self) -> None:
        fields_all = [
            "Question_Type",
            "Subfield",
            "Image_Dependency",
            "Academic_Level",
            "Difficulty",
            "Third-level field",
            "model_correct",
        ]

        dlg = tk.Toplevel(self.root)
        dlg.title("数据集统计（饼图）")
        dlg.geometry("520x220")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text="选择一个字段绘制饼图。可多次绘制，输出到 out-dir。").pack(anchor=tk.W, padx=10, pady=(10, 6))

        box = ttk.Frame(dlg, padding=10)
        box.pack(fill=tk.BOTH, expand=True)

        var = tk.StringVar(value=fields_all[0])
        cmb = ttk.Combobox(box, textvariable=var, state="readonly", values=fields_all, width=28)
        cmb.pack(anchor=tk.W)

        btns = ttk.Frame(dlg, padding=10)
        btns.pack(fill=tk.X)

        def do_draw() -> None:
            field = (var.get() or "").strip()
            if not field:
                messagebox.showwarning("提示", "请选择字段", parent=dlg)
                return
            needs_mc = (field == "model_correct")
            try:
                evaluated_xlsx = None
                if needs_mc:
                    evaluated_xlsx = filedialog.askopenfilename(
                        title="选择模型评测结果 xlsx（包含 model_correct 列）",
                        filetypes=[("Excel", "*.xlsx"), ("All", "*")],
                    )
                    if not evaluated_xlsx:
                        return
                out_path = self._draw_pie(field, evaluated_xlsx_path=evaluated_xlsx)
                self._notify_info("绘制成功", f"饼图已输出：\n{out_path}")
                dlg.destroy()
            except Exception as e:
                messagebox.showerror("绘制失败", f"{type(e).__name__}: {e}", parent=dlg)

        ttk.Button(btns, text="绘制", command=do_draw).pack(side=tk.RIGHT)
        ttk.Button(btns, text="取消", command=dlg.destroy).pack(side=tk.RIGHT, padx=(0, 8))

    @staticmethod
    def _move_listbox_items(src: tk.Listbox, dst: tk.Listbox) -> None:
        sel = list(src.curselection())
        sel.sort()
        items = [src.get(i) for i in sel]
        existing = set(dst.get(0, tk.END))
        for it in items:
            if it not in existing:
                dst.insert(tk.END, it)

    @staticmethod
    def _remove_selected_items(lb: tk.Listbox) -> None:
        sel = list(lb.curselection())
        for i in reversed(sel):
            lb.delete(i)

    @staticmethod
    def _shift_selected(lb: tk.Listbox, delta: int) -> None:
        # Move selected items up/down while preserving relative order
        sel = list(lb.curselection())
        if not sel:
            return
        items = list(lb.get(0, tk.END))
        idx_set = set(sel)

        if delta < 0:
            rng = range(1, len(items))
        else:
            rng = range(len(items) - 2, -1, -1)

        for i in rng:
            j = i + delta
            if j < 0 or j >= len(items):
                continue
            if (i in idx_set) and (j not in idx_set):
                items[i], items[j] = items[j], items[i]
                idx_set.remove(i)
                idx_set.add(j)

        lb.delete(0, tk.END)
        for it in items:
            lb.insert(tk.END, it)
        for i in sorted(idx_set):
            lb.selection_set(i)

    # ---------- Logging ----------
    def _append_log(self, s: str) -> None:
        self.log_text.insert(tk.END, s)
        self.log_text.see(tk.END)

    def _notify_info(self, title: str, msg: str) -> None:
        def _do() -> None:
            try:
                messagebox.showinfo(title, msg)
            except Exception:
                pass
        self.root.after(0, _do)

    def _notify_warn(self, title: str, msg: str) -> None:
        def _do() -> None:
            try:
                messagebox.showwarning(title, msg)
            except Exception:
                pass
        self.root.after(0, _do)

    def _resolve_paths_for_io(self, cfg: Dict[str, Any]) -> Tuple[str, str]:
        """
        Returns (input_xlsx_path, out_dir).
        """
        root_dir = (cfg.get("root") or "").strip()
        input_path = (cfg.get("input") or "").strip()
        if not input_path and root_dir:
            input_path = os.path.join(root_dir, "dataset.xlsx")

        out_dir = (cfg.get("out_dir") or "").strip()
        if not out_dir:
            base_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(__file__)
            out_dir = os.path.join(base_dir, "out_run")
        return input_path, out_dir

    def _load_df_for_plot(self, *, require_model_correct: bool, evaluated_xlsx_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        cfg = self._last_cfg or self._get_cfg()
        input_path, out_dir = self._resolve_paths_for_io(cfg)
        if require_model_correct:
            if not evaluated_xlsx_path:
                raise FileNotFoundError("未选择评测结果 xlsx；请选择包含 model_correct 的 evaluated_*.xlsx 文件。")
            df = pd.read_excel(evaluated_xlsx_path, engine="openpyxl")
            out_dir = os.path.dirname(evaluated_xlsx_path) or out_dir
        else:
            if not input_path:
                raise FileNotFoundError("缺少数据集路径；请先填写 --root（推荐）或 --input。")
            df = pd.read_excel(input_path, engine="openpyxl")
        os.makedirs(out_dir, exist_ok=True)

        # Normalize column names (strip)
        df.columns = [str(c).strip() for c in df.columns]
        return df, out_dir

    def _sanitize_group_series(self, df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(["<missing>"] * len(df))
        s = df[col]
        s = s.astype(str)
        s = s.fillna("<missing>")
        s = s.map(lambda x: x.strip() if isinstance(x, str) else str(x))
        s = s.replace({"": "<missing>", "nan": "<missing>", "None": "<missing>"})
        return s

    def _draw_sankey(self, fields: List[str], *, evaluated_xlsx_path: Optional[str] = None) -> str:
        try:
            import plotly.graph_objects as go  # local import for packaging
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "未安装 plotly，无法绘图。请在运行环境中安装 plotly（pip install plotly），"
                "或如果你在使用打包后的 exe，请在打包机重新执行 build.bat 生成新版本。"
            ) from e

        needs_mc = ("model_correct" in fields)
        df, out_dir = self._load_df_for_plot(require_model_correct=needs_mc, evaluated_xlsx_path=evaluated_xlsx_path)

        if len(fields) < 2:
            raise ValueError("桑基图至少需要选择 2 个字段。")

        # Prepare labels
        series_list = [self._sanitize_group_series(df, f) for f in fields]
        node_labels: List[str] = []
        node_index: Dict[Tuple[str, str], int] = {}

        def _node_id(layer: int, value: str) -> int:
            key = (fields[layer], value)
            if key in node_index:
                return node_index[key]
            idx = len(node_labels)
            node_index[key] = idx
            node_labels.append(f"{fields[layer]}: {value}")
            return idx

        link_source: List[int] = []
        link_target: List[int] = []
        link_value: List[int] = []

        # Build counts for each adjacent pair
        for layer in range(len(fields) - 1):
            a = series_list[layer]
            b = series_list[layer + 1]
            c = pd.DataFrame({"a": a, "b": b}).value_counts().reset_index(name="count")
            for _, row in c.iterrows():
                src = _node_id(layer, str(row["a"]))
                tgt = _node_id(layer + 1, str(row["b"]))
                link_source.append(src)
                link_target.append(tgt)
                link_value.append(int(row["count"]))

        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    node=dict(label=node_labels, pad=12, thickness=16),
                    link=dict(source=link_source, target=link_target, value=link_value),
                )
            ]
        )
        fig.update_layout(title_text="Sankey", font_size=11)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "_".join([f.replace(" ", "_") for f in fields])
        out_path = os.path.join(out_dir, f"sankey_{ts}_{safe}.html")
        fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
        return out_path

    def _draw_pie(self, field: str, *, evaluated_xlsx_path: Optional[str] = None) -> str:
        try:
            import plotly.express as px  # local import for packaging
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "未安装 plotly，无法绘图。请在运行环境中安装 plotly（pip install plotly），"
                "或如果你在使用打包后的 exe，请在打包机重新执行 build.bat 生成新版本。"
            ) from e

        needs_mc = (field == "model_correct")
        df, out_dir = self._load_df_for_plot(require_model_correct=needs_mc, evaluated_xlsx_path=evaluated_xlsx_path)

        s = self._sanitize_group_series(df, field)
        vc = s.value_counts().reset_index()
        vc.columns = [field, "count"]
        fig = px.pie(vc, names=field, values="count", title=f"Pie: {field}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"pie_{ts}_{field}.html")
        fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
        return out_path

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
