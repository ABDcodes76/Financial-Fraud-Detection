import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Ensure clean UTF-8 console output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent

def find_git():
    git = shutil.which('git')
    if git:
        return git
    candidates = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        r"C:\Users\dell\AppData\Local\Programs\Git\cmd\git.exe",
        r"C:\Users\dell\AppData\Local\Programs\MinGit\cmd\git.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def run_git_cmd(git_exe, args, cwd=PROJECT_ROOT):
    cmd = [git_exe] + args
    print(f"-> Running: git {' '.join(args)}")
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr and res.returncode != 0:
        print("Git Info / Notice:", res.stderr.strip())
    return res.returncode == 0

def main():
    git_exe = find_git()
    if not git_exe:
        print("❌ Git was not found on your system.")
        print("Please install Git for Windows from: https://git-scm.com/download/win")
        return

    print("=" * 60)
    print("FINSEC AI — Automatic GitHub Sync Utility")
    print("=" * 60)

    # 1. Initialize repository if needed
    if not (PROJECT_ROOT / ".git").exists():
        print("\n[1/4] Initializing local Git repository...")
        run_git_cmd(git_exe, ["init"])
        run_git_cmd(git_exe, ["branch", "-M", "main"])
    else:
        print("\n[1/4] Local Git repository detected.")

    # 2. Check remote origin
    DEFAULT_REPO = "https://github.com/ABDcodes76/Financial-Fraud-Detection.git"
    res = subprocess.run([git_exe, "remote", "-v"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    if "origin" not in res.stdout:
        print(f"\nSetting remote origin to: {DEFAULT_REPO}")
        run_git_cmd(git_exe, ["remote", "add", "origin", DEFAULT_REPO])
        print("✅ Remote origin configured successfully!")

    # 3. Stage changes
    print("\n[2/4] Staging modified and new files...")
    run_git_cmd(git_exe, ["add", "."])

    # 4. Commit
    commit_msg = sys.argv[1] if len(sys.argv) > 1 else f"Update FINSEC AI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    print(f"\n[3/4] Committing changes with message: '{commit_msg}'...")
    run_git_cmd(git_exe, ["commit", "-m", commit_msg])

    # 5. Push to GitHub
    print("\n[4/4] Pushing changes to GitHub (main branch)...")
    success = run_git_cmd(git_exe, ["push", "-u", "origin", "main"])
    if success:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS: Your latest code is now live on GitHub!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("ℹ️ If this is your first push, Git might prompt you in a browser")
        print("window to Sign in with GitHub. Simply click 'Authorize'.")
        print("=" * 60)

if __name__ == "__main__":
    main()
