#!/bin/sh
# B70 Launcher rootless installer (POSIX sh).
#
# Installs the inspectable source bundle into the current user's home only:
#   app files    ${XDG_DATA_HOME:-~/.local/share}/b70-launcher
#   commands     ~/.local/bin/b70-launcher (GUI/daemon) and ~/.local/bin/b70 (CLI)
#   menu entry   ${XDG_DATA_HOME:-~/.local/share}/applications/b70-launcher.desktop
#   icons        ${XDG_DATA_HOME:-~/.local/share}/icons/hicolor/*/apps/b70-launcher.*
#
# An existing user-level installation is replaced in place. User state
# (usage history, logs, settings overrides, server re-adoption data) lives
# under ${XDG_STATE_HOME:-~/.local/state}/b70-launcher and is never touched.
# Nothing outside the home directory is written, and no engine, model,
# Docker image, or power setting is changed.
set -eu

say()  { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
sedesc() { printf '%s' "$1" | sed 's/[&|\\]/\\&/g'; }

umask 022

case ${1:-""} in
    "") ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) die "this installer takes no options (got: $1)" ;;
esac

if [ "$(id -u)" = "0" ]; then
    die "this installer is rootless by design: running it as root would create files your user cannot manage. Re-run as your normal user, without sudo."
fi
[ -n "${HOME:-}" ] || die "HOME is not set; cannot locate ~/.local."

# --- prerequisites ---------------------------------------------------------
py=$(command -v python3 || true)
[ -n "$py" ] || die "python3 not found in PATH. Install Python 3.9+ with your package manager (Debian/Ubuntu: sudo apt install python3; Fedora: sudo dnf install python3) and re-run."
"$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
    || die "python3 is too old ($("$py" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown)). B70 Launcher needs Python 3.9+; install a newer python3 and re-run."

# --- source bundle sanity ----------------------------------------------------
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
for f in launcher.py cli.py webwindow.py appwindow.py recipes.json settings.json README.md \
         web/index.html packaging/b70-launcher.desktop; do
    [ -f "$root/$f" ] || die "missing bundle file: $f. Run this script as packaging/install.sh inside the extracted b70-launcher-*-linux-source directory."
done
set -- "$root"/patches/*.py
[ -f "$1" ] || die "missing bundle files: patches/*.py - the source archive looks incomplete."
set -- "$root"/web/assets/*
[ -f "$1" ] || die "missing bundle files: web/assets/* - the source archive looks incomplete."

# --- target locations --------------------------------------------------------
data_home=${XDG_DATA_HOME:-$HOME/.local/share}
state_dir=${XDG_STATE_HOME:-$HOME/.local/state}/b70-launcher
app=$data_home/b70-launcher
bin=$HOME/.local/bin
desktop_dir=$data_home/applications
icons_root=$data_home/icons/hicolor

# --- optional-dependency notes (non-fatal) ------------------------------------
webkit_typelibs=false
for d in /usr/lib/x86_64-linux-gnu/girepository-1.0 \
         /usr/lib64/girepository-1.0 /usr/lib/girepository-1.0 \
         /usr/lib/aarch64-linux-gnu/girepository-1.0; do
    if [ -f "$d/Gtk-3.0.typelib" ] && [ -f "$d/WebKit2-4.1.typelib" ]; then
        webkit_typelibs=true
        break
    fi
    if [ -f "$d/Gtk-3.0.typelib" ] && [ -f "$d/WebKit2-4.0.typelib" ]; then
        webkit_typelibs=true
        break
    fi
    if [ -f "$d/Gtk-4.0.typelib" ] && [ -f "$d/WebKit-6.0.typelib" ]; then
        webkit_typelibs=true
        break
    fi
done
gi_ok=true
"$py" -c 'import gi' >/dev/null 2>&1 || gi_ok=false
if ! $webkit_typelibs || ! $gi_ok; then
    warn "python3-gi and/or a WebKitGTK typelib (4.1, 4.0, or GTK4 WebKit-6.0) not found - the native app window will be unavailable and the launcher will fall back to opening in your default browser. For the native window install e.g. 'gir1.2-webkit2-4.1 python3-gi' (Debian/Ubuntu) or 'webkit2gtk4.1' + 'python3-gobject' (Fedora)."
fi
command -v docker >/dev/null 2>&1 \
    || warn "docker CLI not found - the UI installs and opens fine, but no GPU engine can launch until Docker with Intel GPU support is available."

# --- install -----------------------------------------------------------------
if [ -e "$app" ]; then
    case $app in
        */b70-launcher) ;;
        *) die "refusing to remove unexpected path: $app" ;;
    esac
    say "Replacing existing installation at $app (state in $state_dir is preserved)."
    rm -rf -- "$app"
fi

mkdir -p "$app/web/assets" "$app/patches" "$bin" "$desktop_dir"
cp "$root/launcher.py" "$root/cli.py" "$root/webwindow.py" "$root/appwindow.py" \
   "$root/recipes.json" "$root/settings.json" "$root/README.md" "$app/"
cp "$root"/patches/*.py "$app/patches/"
cp -R "$root"/patches/ssu-b70-b8w4 "$app/patches/" 2>/dev/null || true
cp "$root/web/index.html" "$app/web/"
cp -R "$root/web/assets/." "$app/web/assets/"
if [ -d "$root/docs" ]; then
    mkdir -p "$app/docs"
    cp "$root"/docs/*.md "$app/docs/" 2>/dev/null || true
fi
if [ -f "$root/packaging/uninstall.sh" ]; then
    cp "$root/packaging/uninstall.sh" "$app/uninstall.sh"
    chmod 755 "$app/uninstall.sh"
fi

cat > "$bin/b70-launcher" <<EOF
#!/bin/sh
# B70 Launcher wrapper - generated by packaging/install.sh
exec python3 "$app/launcher.py" "\$@"
EOF
chmod 755 "$bin/b70-launcher"

cat > "$bin/b70" <<EOF
#!/bin/sh
# B70 Launcher wrapper - generated by packaging/install.sh
# b70: headless CLI client for the B70 launcher daemon
exec python3 "$app/cli.py" "\$@"
EOF
chmod 755 "$bin/b70"

for png in "$root"/web/assets/b70-launcher-*.png; do
    [ -f "$png" ] || continue
    size=${png##*b70-launcher-}; size=${size%.png}
    case $size in ''|*[!0-9]*) continue ;; esac
    mkdir -p "$icons_root/${size}x${size}/apps"
    cp "$png" "$icons_root/${size}x${size}/apps/b70-launcher.png"
done
if [ -f "$root/web/assets/b70-launcher.svg" ]; then
    mkdir -p "$icons_root/scalable/apps"
    cp "$root/web/assets/b70-launcher.svg" "$icons_root/scalable/apps/b70-launcher.svg"
fi

exec_path=$bin/b70-launcher
# Exec is quoted so an absolute path containing spaces still launches;
# TryExec takes a raw path per the desktop entry spec.
sed -e "s|^Exec=.*|Exec=\"$(sedesc "$exec_path")\"|" \
    -e "s|^TryExec=.*|TryExec=$(sedesc "$exec_path")|" \
    "$root/packaging/b70-launcher.desktop" > "$desktop_dir/b70-launcher.desktop"
chmod 644 "$desktop_dir/b70-launcher.desktop"

# --- best-effort desktop integration ------------------------------------------
if command -v desktop-file-validate >/dev/null 2>&1; then
    desktop-file-validate "$desktop_dir/b70-launcher.desktop" \
        || warn "desktop entry failed validation; the menu entry may not appear correctly."
fi
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$desktop_dir" >/dev/null 2>&1 || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t "$icons_root" >/dev/null 2>&1 || true
fi

# --- PATH check ---------------------------------------------------------------
case :${PATH:-}: in
    *":$bin:"*) ;;
    *) warn "$bin is not in PATH for this shell.
       The application menu entry works regardless (it uses an absolute path).
       To run 'b70-launcher' from a terminal, add this line to ~/.profile or your
       shell's rc file, then open a new terminal:
           export PATH=\"\$HOME/.local/bin:\$PATH\"
       Or invoke the full path directly: $bin/b70-launcher" ;;
esac

# --- summary ------------------------------------------------------------------
say "Installed:"
say "  app files : $app"
say "  commands  : $bin/b70-launcher (UI/daemon) + $bin/b70 (headless CLI)"
say "  menu entry: $desktop_dir/b70-launcher.desktop"
say "  icons     : $icons_root/*/apps/b70-launcher.*"
say "State (usage history, logs, settings overrides) lives in $state_dir and is preserved across reinstalls."
if [ -f "$app/uninstall.sh" ]; then
    say "Uninstall : sh $app/uninstall.sh   (keeps state; --purge also removes it)"
else
    say "Uninstall : remove $app, $bin/b70-launcher, $desktop_dir/b70-launcher.desktop, and the hicolor icon files (keeps $state_dir)"
fi
say "No engine, model, Docker image, or power setting changed."
