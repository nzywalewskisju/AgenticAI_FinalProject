# gui.py
# Desktop GUI entry point for the HR Policy Assistant.
# Run with: python gui.py
# Launches a login dialog first, then the main application window.
# Login dialog handles: account creation, login, forgot password flow.
# Main window contains:
#   - File picker for uploading HR documents from anywhere on disk
#   - Document panel showing all ingested documents with per-doc clear buttons
#   - User profile panel showing remembered facts with ability to delete individual facts
#   - Query input field (Enter key or Submit button)
#   - Scrollable answer display with citations
#   - Status indicator: idle / ingesting / thinking / ready / error
#   - Clear Session button that resets conversation memory only
# All inference and storage is local — no external API calls at runtime.
# Errors from Ollama and ChromaDB are surfaced in the window, not the terminal.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import uuid
import json
import os
import bcrypt
from config import USERS_FILE, SECURITY_QUESTIONS


# ── Auth helpers ───────────────────────────────────────────────────────────────

def _load_users() -> list:
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: list) -> None:
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def authenticate_user(username: str, password: str) -> dict | None:
    users = _load_users()
    user = next((u for u in users if u["username"].lower() == username.lower()), None)
    if not user or not user.get("password_hash"):
        return None
    if bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return user
    return None


def create_user(username: str, password: str, security_question: str, security_answer: str) -> dict | None:
    users = _load_users()
    if any(u["username"].lower() == username.lower() for u in users):
        return None  # Username taken
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    answer_hash = bcrypt.hashpw(security_answer.strip().lower().encode(), bcrypt.gensalt()).decode()
    user = {
        "user_id": str(uuid.uuid4()),
        "username": username,
        "password_hash": password_hash,
        "security_question": security_question,
        "security_answer_hash": answer_hash,
        "created_at": __import__("datetime").datetime.utcnow().isoformat()
    }
    users.append(user)
    _save_users(users)
    return user


def verify_security_answer(username: str, answer: str) -> bool:
    users = _load_users()
    user = next((u for u in users if u["username"].lower() == username.lower()), None)
    if not user:
        return False
    return bcrypt.checkpw(answer.strip().lower().encode(), user["security_answer_hash"].encode())


def update_password(username: str, new_password: str) -> bool:
    users = _load_users()
    user = next((u for u in users if u["username"].lower() == username.lower()), None)
    if not user:
        return False
    user["password_hash"] = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    _save_users(users)
    return True


# ── Colour palette ─────────────────────────────────────────────────────────────

BG          = "#0f1117"
BG_PANEL    = "#1a1d27"
BG_INPUT    = "#22263a"
ACCENT      = "#4f8ef7"
ACCENT_DIM  = "#2d4f8a"
TEXT        = "#e8eaf0"
TEXT_DIM    = "#7b8099"
TEXT_CITE   = "#a0c4ff"
SUCCESS     = "#4caf82"
WARNING     = "#f0a040"
ERROR       = "#e05555"
BORDER      = "#2e3245"
FONT_MAIN   = ("Helvetica Neue", 12)
FONT_SMALL  = ("Helvetica Neue", 10)
FONT_MONO   = ("Courier New", 11)
FONT_TITLE  = ("Helvetica Neue", 18, "bold")
FONT_LABEL  = ("Helvetica Neue", 11, "bold")


# ── Login window ───────────────────────────────────────────────────────────────

class LoginWindow:
    def __init__(self, on_success):
        self.on_success = on_success
        self.root = tk.Tk()
        self.root.title("HR Policy Assistant")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self._center(420, 520)
        self._build_login_view()
        self.root.mainloop()

    def _center(self, w, h):
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _clear(self):
        for w in self.root.winfo_children():
            w.destroy()

    def _label(self, parent, text, font=FONT_MAIN, fg=TEXT, **kwargs):
        return tk.Label(parent, text=text, font=font, fg=fg, bg=BG, **kwargs)

    def _entry(self, parent, show=None):
        e = tk.Entry(parent, font=FONT_MAIN, fg=TEXT, bg=BG_INPUT,
                     insertbackground=TEXT, relief="flat",
                     highlightthickness=1, highlightcolor=ACCENT,
                     highlightbackground=BORDER, show=show)
        return e

    def _btn(self, parent, text, command, primary=True):
        bg = ACCENT if primary else BG_INPUT
        return tk.Button(parent, text=text, command=command,
                         font=FONT_LABEL, fg=TEXT, bg=bg,
                         activebackground=ACCENT_DIM, activeforeground=TEXT,
                         relief="flat", cursor="hand2", pady=8)

    def _build_login_view(self):
        self._clear()
        pad = {"padx": 40, "pady": 6}

        self._label(self.root, "HR Policy Assistant", font=FONT_TITLE).pack(pady=(40, 4))
        self._label(self.root, "Sign in to continue", fg=TEXT_DIM, font=FONT_SMALL).pack(pady=(0, 24))

        self._label(self.root, "Username").pack(anchor="w", **pad)
        self.username_var = tk.StringVar()
        self._entry(self.root).pack(fill="x", **pad)
        self.username_entry = self.root.winfo_children()[-1]
        self.username_entry.config(textvariable=self.username_var)

        self._label(self.root, "Password").pack(anchor="w", **pad)
        self.password_var = tk.StringVar()
        self._entry(self.root, show="•").pack(fill="x", **pad)
        self.password_entry = self.root.winfo_children()[-1]
        self.password_entry.config(textvariable=self.password_var)
        self.password_entry.bind("<Return>", lambda e: self._do_login())

        self.error_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.error_var, font=FONT_SMALL,
                 fg=ERROR, bg=BG).pack(pady=4)

        self._btn(self.root, "Sign In", self._do_login).pack(fill="x", padx=40, pady=4)

        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(pady=8)
        tk.Button(bottom, text="Create Account", command=self._build_create_view,
                  font=FONT_SMALL, fg=ACCENT, bg=BG, relief="flat",
                  cursor="hand2", activeforeground=TEXT, activebackground=BG).pack(side="left", padx=8)
        tk.Button(bottom, text="Forgot Password?", command=self._build_forgot_view,
                  font=FONT_SMALL, fg=TEXT_DIM, bg=BG, relief="flat",
                  cursor="hand2", activeforeground=TEXT, activebackground=BG).pack(side="left", padx=8)

    def _do_login(self):
        username = self.username_var.get().strip()
        password = self.password_var.get()
        if not username or not password:
            self.error_var.set("Please enter username and password.")
            return
        user = authenticate_user(username, password)
        if user:
            self.root.destroy()
            self.on_success(user)
        else:
            self.error_var.set("Incorrect username or password.")

    def _build_create_view(self):
        self._clear()
        pad = {"padx": 40, "pady": 4}

        self._label(self.root, "Create Account", font=FONT_TITLE).pack(pady=(32, 16))

        fields = {}
        for label, show in [("Username", None), ("Password", "•"), ("Confirm Password", "•")]:
            self._label(self.root, label).pack(anchor="w", **pad)
            var = tk.StringVar()
            e = self._entry(self.root, show=show)
            e.config(textvariable=var)
            e.pack(fill="x", **pad)
            fields[label] = var

        self._label(self.root, "Security Question").pack(anchor="w", **pad)
        sq_var = tk.StringVar(value=SECURITY_QUESTIONS[0])
        sq_menu = ttk.Combobox(self.root, textvariable=sq_var,
                               values=SECURITY_QUESTIONS,
                               font=FONT_SMALL, state="readonly")
        sq_menu.pack(fill="x", **pad)

        self._label(self.root, "Security Answer").pack(anchor="w", **pad)
        sa_var = tk.StringVar()
        self._entry(self.root).pack(fill="x", **pad)
        sa_entry = self.root.winfo_children()[-1]
        sa_entry.config(textvariable=sa_var)

        err_var = tk.StringVar()
        tk.Label(self.root, textvariable=err_var, font=FONT_SMALL,
                 fg=ERROR, bg=BG).pack(pady=2)

        def do_create():
            u = fields["Username"].get().strip()
            p = fields["Password"].get()
            c = fields["Confirm Password"].get()
            sq = sq_var.get()
            sa = sa_var.get().strip()
            if not all([u, p, c, sa]):
                err_var.set("All fields are required.")
                return
            if p != c:
                err_var.set("Passwords do not match.")
                return
            if len(p) < 6:
                err_var.set("Password must be at least 6 characters.")
                return
            user = create_user(u, p, sq, sa)
            if not user:
                err_var.set("Username already taken.")
                return
            self.root.destroy()
            self.on_success(user)

        self._btn(self.root, "Create Account", do_create).pack(fill="x", padx=40, pady=8)
        self._btn(self.root, "← Back to Sign In", self._build_login_view, primary=False).pack(fill="x", padx=40)

    def _build_forgot_view(self):
        self._clear()
        pad = {"padx": 40, "pady": 6}
        self._label(self.root, "Reset Password", font=FONT_TITLE).pack(pady=(40, 16))

        self._label(self.root, "Username").pack(anchor="w", **pad)
        u_var = tk.StringVar()
        self._entry(self.root).pack(fill="x", **pad)
        self.root.winfo_children()[-1].config(textvariable=u_var)

        question_var = tk.StringVar(value="Enter your username above first.")
        tk.Label(self.root, textvariable=question_var, font=FONT_SMALL,
                 fg=TEXT_DIM, bg=BG, wraplength=320).pack(pady=4)

        self._label(self.root, "Security Answer").pack(anchor="w", **pad)
        a_var = tk.StringVar()
        self._entry(self.root).pack(fill="x", **pad)
        self.root.winfo_children()[-1].config(textvariable=a_var)

        self._label(self.root, "New Password").pack(anchor="w", **pad)
        p_var = tk.StringVar()
        self._entry(self.root, show="•").pack(fill="x", **pad)
        self.root.winfo_children()[-1].config(textvariable=p_var)

        err_var = tk.StringVar()
        tk.Label(self.root, textvariable=err_var, font=FONT_SMALL,
                 fg=ERROR, bg=BG).pack(pady=2)

        def lookup_question(*_):
            username = u_var.get().strip()
            if not username:
                return
            users = _load_users()
            user = next((u for u in users if u["username"].lower() == username.lower()), None)
            if user:
                question_var.set(f"Security question: {user['security_question']}")
            else:
                question_var.set("No account found with that username.")

        u_var.trace_add("write", lookup_question)

        def do_reset():
            username = u_var.get().strip()
            answer = a_var.get().strip()
            new_pw = p_var.get()
            if not all([username, answer, new_pw]):
                err_var.set("All fields are required.")
                return
            if not verify_security_answer(username, answer):
                err_var.set("Incorrect security answer.")
                return
            if len(new_pw) < 6:
                err_var.set("Password must be at least 6 characters.")
                return
            update_password(username, new_pw)
            messagebox.showinfo("Success", "Password updated. Please sign in.")
            self._build_login_view()

        self._btn(self.root, "Reset Password", do_reset).pack(fill="x", padx=40, pady=8)
        self._btn(self.root, "← Back to Sign In", self._build_login_view, primary=False).pack(fill="x", padx=40)


# ── Main application window ────────────────────────────────────────────────────

class MainWindow:
    def __init__(self, user: dict):
        self.user = user
        self.user_id = user["user_id"]
        self.username = user["username"]
        self.session_id = str(uuid.uuid4())
        self.selected_files = []

        self.root = tk.Tk()
        self.root.title(f"HR Policy Assistant — {self.username}")
        self.root.configure(bg=BG)
        self.root.minsize(960, 680)
        self._center(1100, 740)
        self._build()
        self._refresh_documents()
        self._refresh_profile()
        self.root.mainloop()

    def _center(self, w, h):
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _build(self):
        # ── Top bar ────────────────────────────────────────────────────────────
        topbar = tk.Frame(self.root, bg=BG_PANEL, pady=12)
        topbar.pack(fill="x")
        tk.Label(topbar, text="HR Policy Assistant", font=FONT_TITLE,
                 fg=TEXT, bg=BG_PANEL).pack(side="left", padx=20)
        tk.Label(topbar, text=f"Signed in as {self.username}",
                 font=FONT_SMALL, fg=TEXT_DIM, bg=BG_PANEL).pack(side="right", padx=20)

        self.status_var = tk.StringVar(value="● Ready")
        self.status_label = tk.Label(topbar, textvariable=self.status_var,
                                     font=FONT_SMALL, fg=SUCCESS, bg=BG_PANEL)
        self.status_label.pack(side="right", padx=12)

        # ── Main layout ────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        # Left sidebar
        sidebar = tk.Frame(body, bg=BG_PANEL, width=260)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Right: chat area
        chat_area = tk.Frame(body, bg=BG)
        chat_area.pack(side="left", fill="both", expand=True)
        self._build_chat(chat_area)

    def _build_sidebar(self, parent):
        # ── Documents section ──────────────────────────────────────────────────
        tk.Label(parent, text="Documents", font=FONT_LABEL,
                 fg=TEXT, bg=BG_PANEL).pack(anchor="w", padx=16, pady=(16, 4))

        file_btn = tk.Button(parent, text="+ Add Documents",
                             command=self._pick_files,
                             font=FONT_SMALL, fg=TEXT, bg=ACCENT,
                             activebackground=ACCENT_DIM, relief="flat",
                             cursor="hand2", pady=6)
        file_btn.pack(fill="x", padx=16, pady=4)

        self.ingest_btn = tk.Button(parent, text="Ingest Selected Files",
                                    command=self._run_ingestion,
                                    font=FONT_SMALL, fg=TEXT, bg=BG_INPUT,
                                    activebackground=ACCENT_DIM, relief="flat",
                                    cursor="hand2", pady=6, state="disabled")
        self.ingest_btn.pack(fill="x", padx=16, pady=2)

        self.selected_label = tk.Label(parent, text="No files selected.",
                                       font=FONT_SMALL, fg=TEXT_DIM,
                                       bg=BG_PANEL, wraplength=220, justify="left")
        self.selected_label.pack(anchor="w", padx=16, pady=4)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=16, pady=8)
        tk.Label(parent, text="Loaded Documents", font=FONT_LABEL,
                 fg=TEXT, bg=BG_PANEL).pack(anchor="w", padx=16, pady=(0, 4))

        self.docs_frame = tk.Frame(parent, bg=BG_PANEL)
        self.docs_frame.pack(fill="x", padx=16)

        clear_all_btn = tk.Button(parent, text="Clear All Documents",
                                  command=self._clear_all_docs,
                                  font=FONT_SMALL, fg=ERROR, bg=BG_PANEL,
                                  relief="flat", cursor="hand2",
                                  activeforeground=TEXT, activebackground=BG_PANEL)
        clear_all_btn.pack(anchor="w", padx=16, pady=4)

        # ── Profile section ────────────────────────────────────────────────────
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=16, pady=8)
        tk.Label(parent, text="My Profile", font=FONT_LABEL,
                 fg=TEXT, bg=BG_PANEL).pack(anchor="w", padx=16, pady=(0, 4))

        self.profile_frame = tk.Frame(parent, bg=BG_PANEL)
        self.profile_frame.pack(fill="x", padx=16)

        # ── Session controls ───────────────────────────────────────────────────
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=16, pady=8)
        tk.Button(parent, text="Clear Session",
                  command=self._clear_session,
                  font=FONT_SMALL, fg=TEXT_DIM, bg=BG_PANEL,
                  relief="flat", cursor="hand2",
                  activeforeground=TEXT, activebackground=BG_PANEL
                  ).pack(anchor="w", padx=16, pady=2)

    def _build_chat(self, parent):
        # Answer display
        answer_frame = tk.Frame(parent, bg=BG)
        answer_frame.pack(fill="both", expand=True, padx=16, pady=(16, 8))

        self.answer_display = scrolledtext.ScrolledText(
            answer_frame,
            font=FONT_MONO, fg=TEXT, bg=BG_INPUT,
            relief="flat", wrap="word",
            state="disabled", padx=16, pady=16,
            highlightthickness=1, highlightbackground=BORDER
        )
        self.answer_display.pack(fill="both", expand=True)

        # Configure text tags for formatting
        self.answer_display.tag_configure("user", foreground=ACCENT, font=("Helvetica Neue", 11, "bold"))
        self.answer_display.tag_configure("assistant", foreground=TEXT, font=FONT_MONO)
        self.answer_display.tag_configure("citation", foreground=TEXT_CITE, font=FONT_SMALL)
        self.answer_display.tag_configure("status", foreground=TEXT_DIM, font=FONT_SMALL)
        self.answer_display.tag_configure("fact", foreground=SUCCESS, font=FONT_SMALL)
        self.answer_display.tag_configure("error", foreground=ERROR, font=FONT_SMALL)
        self.answer_display.tag_configure("separator", foreground=BORDER)

        # Input area
        input_frame = tk.Frame(parent, bg=BG, pady=8)
        input_frame.pack(fill="x", padx=16, pady=(0, 16))

        self.query_input = tk.Text(input_frame, font=FONT_MAIN,
                                   fg=TEXT, bg=BG_INPUT,
                                   insertbackground=TEXT, relief="flat",
                                   highlightthickness=1, highlightcolor=ACCENT,
                                   highlightbackground=BORDER,
                                   height=3, padx=12, pady=10, wrap="word")
        self.query_input.pack(side="left", fill="both", expand=True)
        self.query_input.bind("<Return>", self._on_enter)
        self.query_input.bind("<Shift-Return>", lambda e: None)

        submit_btn = tk.Button(input_frame, text="Send",
                               command=self._submit_query,
                               font=FONT_LABEL, fg=TEXT, bg=ACCENT,
                               activebackground=ACCENT_DIM,
                               relief="flat", cursor="hand2",
                               padx=20, pady=10)
        submit_btn.pack(side="left", padx=(8, 0))

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _on_enter(self, event):
        if not event.state & 0x1:   # Shift not held
            self._submit_query()
            return "break"

    def _pick_files(self):
        files = filedialog.askopenfilenames(
            title="Select HR Policy Documents",
            filetypes=[("Supported files", "*.pdf *.docx"), ("PDF", "*.pdf"), ("Word", "*.docx")]
        )
        if files:
            self.selected_files = list(files)
            names = [os.path.basename(f) for f in files]
            display = ", ".join(names[:2])
            if len(names) > 2:
                display += f" (+{len(names) - 2} more)"
            self.selected_label.config(text=display, fg=TEXT)
            self.ingest_btn.config(state="normal")

    def _run_ingestion(self):
        if not self.selected_files:
            return

        self._set_status("● Ingesting...", WARNING)
        self.ingest_btn.config(state="disabled")
        self._append_chat("status", f"Ingesting {len(self.selected_files)} document(s)...\n")

        def ingest_thread():
            try:
                from src.ingestion.embedder import run_ingestion_pipeline
                result = run_ingestion_pipeline(self.selected_files, self.user_id)
                self.root.after(0, lambda: self._on_ingestion_complete(result))
            except Exception as e:
                self.root.after(0, lambda: self._on_ingestion_error(str(e)))

        threading.Thread(target=ingest_thread, daemon=True).start()

    def _on_ingestion_complete(self, result):
        self._set_status("● Ready", SUCCESS)
        self._append_chat("status", f"✓ Ingested {result['chunks_stored']} chunks from {result['files_processed']} document(s).\n\n")
        self.selected_files = []
        self.selected_label.config(text="No files selected.", fg=TEXT_DIM)
        self._refresh_documents()

    def _on_ingestion_error(self, error: str):
        self._set_status("● Error", ERROR)
        self._append_chat("error", f"✗ Ingestion failed: {error}\n\n")
        self.ingest_btn.config(state="normal")

    def _submit_query(self):
        query = self.query_input.get("1.0", "end").strip()
        if not query:
            return
        self.query_input.delete("1.0", "end")
        self._append_chat("user", f"You: {query}\n")
        self._set_status("● Thinking...", WARNING)

        def query_thread():
            try:
                from src.agents.orchestrator import run_orchestrator
                result = run_orchestrator(query, user_id=self.user_id, session_id=self.session_id)
                self.root.after(0, lambda: self._on_query_complete(result))
            except Exception as e:
                self.root.after(0, lambda: self._on_query_error(str(e)))

        threading.Thread(target=query_thread, daemon=True).start()
        self._stream_loading_steps()

    def _stream_loading_steps(self):
        """Shows animated loading messages while the agent is working."""
        steps = [
            "Classifying your query...",
            "Checking policy coverage...",
            "Retrieving relevant sections...",
            "Reasoning about your situation...",
            "Reviewing answer quality...",
            "Running compliance check...",
            "Finalizing response..."
        ]
        self._loading_index = 0
        self._loading_tag_start = None

        def show_next_step():
            if self._loading_index < len(steps):
                step = steps[self._loading_index]
                self._replace_loading_step(f"  ⟳ {step}\n")
                self._loading_index += 1
                self._loading_after_id = self.root.after(2200, show_next_step)

        self._loading_after_id = self.root.after(300, show_next_step)

    def _replace_loading_step(self, text: str):
        self.answer_display.config(state="normal")
        if self._loading_tag_start:
            self.answer_display.delete(self._loading_tag_start, "end")
        self._loading_tag_start = self.answer_display.index("end-1c")
        self.answer_display.insert("end", text, "status")
        self.answer_display.see("end")
        self.answer_display.config(state="disabled")

    def _cancel_loading(self):
        if hasattr(self, "_loading_after_id"):
            self.root.after_cancel(self._loading_after_id)
        if hasattr(self, "_loading_tag_start") and self._loading_tag_start:
            self.answer_display.config(state="normal")
            self.answer_display.delete(self._loading_tag_start, "end")
            self.answer_display.config(state="disabled")
            self._loading_tag_start = None

    def _on_query_complete(self, result: dict):
        self._cancel_loading()
        self._set_status("● Ready", SUCCESS)

        answer = result.get("answer", "No response.")
        # Split citations from main answer for separate formatting
        if "---\n**Sources:**" in answer:
            parts = answer.split("---\n**Sources:**")
            self._append_chat("assistant", f"\n{parts[0].strip()}\n")
            self._append_chat("citation", f"\nSources:{parts[1]}\n")
        else:
            self._append_chat("assistant", f"\n{answer}\n")

        # New profile facts notification
        for fact in result.get("new_profile_facts", []):
            self._append_chat("fact", f"✓ Profile updated: {fact}\n")

        self._append_chat("separator", "─" * 60 + "\n\n")
        self._refresh_profile()

    def _on_query_error(self, error: str):
        self._cancel_loading()
        self._set_status("● Error", ERROR)
        self._append_chat("error", f"✗ Error: {error}\n\n")

    def _clear_session(self):
        from src.memory.session import session_memory
        session_memory.clear_session(self.session_id)
        self.session_id = str(uuid.uuid4())
        self._append_chat("status", "Session cleared. Starting fresh.\n\n")

    def _clear_all_docs(self):
        if messagebox.askyesno("Clear Documents",
                               "Remove all documents? This cannot be undone."):
            from src.memory.registry import clear_all_documents
            count = clear_all_documents(self.user_id)
            self._refresh_documents()
            self._append_chat("status", f"Cleared {count} document(s).\n\n")

    # ── Sidebar refresh ────────────────────────────────────────────────────────

    def _refresh_documents(self):
        for w in self.docs_frame.winfo_children():
            w.destroy()
        from src.tools.document import get_registry
        registry = get_registry(self.user_id)
        if not registry:
            tk.Label(self.docs_frame, text="No documents loaded.",
                     font=FONT_SMALL, fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w")
            return
        for record in registry:
            row = tk.Frame(self.docs_frame, bg=BG_PANEL)
            row.pack(fill="x", pady=2)
            name = record.get("file_name", "Unknown")
            chunks = record.get("chunk_count", 0)
            tk.Label(row, text=f"📄 {name[:22]}{'...' if len(name) > 22 else ''}",
                     font=FONT_SMALL, fg=TEXT, bg=BG_PANEL,
                     anchor="w").pack(side="left", fill="x", expand=True)
            tk.Label(row, text=f"{chunks}c",
                     font=FONT_SMALL, fg=TEXT_DIM, bg=BG_PANEL).pack(side="left")
            file_path = record.get("file_path", "")
            tk.Button(row, text="✕", command=lambda p=file_path: self._remove_doc(p),
                      font=FONT_SMALL, fg=ERROR, bg=BG_PANEL,
                      relief="flat", cursor="hand2",
                      activeforeground=TEXT, activebackground=BG_PANEL,
                      padx=4).pack(side="right")

    def _remove_doc(self, file_path: str):
        from src.tools.document import remove_from_registry
        remove_from_registry(self.user_id, file_path)
        self._refresh_documents()

    def _refresh_profile(self):
        for w in self.profile_frame.winfo_children():
            w.destroy()
        from src.memory.profile import load_profile
        profile = load_profile(self.user_id)
        facts = profile.get("facts", {})
        if not facts:
            tk.Label(self.profile_frame, text="No profile facts yet.",
                     font=FONT_SMALL, fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w")
            return
        for key, value in facts.items():
            row = tk.Frame(self.profile_frame, bg=BG_PANEL)
            row.pack(fill="x", pady=1)
            label = key.replace("_", " ").title()
            tk.Label(row, text=f"{label}: {value}",
                     font=FONT_SMALL, fg=TEXT, bg=BG_PANEL,
                     anchor="w", wraplength=180).pack(side="left", fill="x", expand=True)
            tk.Button(row, text="✕", command=lambda k=key: self._delete_fact(k),
                      font=FONT_SMALL, fg=ERROR, bg=BG_PANEL,
                      relief="flat", cursor="hand2",
                      activeforeground=TEXT, activebackground=BG_PANEL,
                      padx=4).pack(side="right")

    def _delete_fact(self, fact_key: str):
        from src.memory.profile import delete_profile_fact
        delete_profile_fact(self.user_id, fact_key)
        self._refresh_profile()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _append_chat(self, tag: str, text: str):
        self.answer_display.config(state="normal")
        self.answer_display.insert("end", text, tag)
        self.answer_display.see("end")
        self.answer_display.config(state="disabled")

    def _set_status(self, text: str, color: str):
        self.status_var.set(text)
        self.status_label.config(fg=color)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    def on_login_success(user: dict):
        MainWindow(user)

    LoginWindow(on_success=on_login_success)


if __name__ == "__main__":
    main()