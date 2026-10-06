#!/bin/sh
# Install this inspected source bundle in the current user's home only.
set -eu
command -v python3 >/dev/null 2>&1 || { echo 'Python 3.9+ is required' >&2; exit 1; }
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || { echo 'Python 3.9+ is required' >&2; exit 1; }
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
app="${XDG_DATA_HOME:-$HOME/.local/share}/b70-launcher"
bin="$HOME/.local/bin"
desktop="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$app/web/assets" "$app/patches" "$bin" "$desktop"
cp "$root/launcher.py" "$root/webwindow.py" "$root/appwindow.py" "$root/recipes.json" "$root/settings.json" "$root/README.md" "$app/"
cp "$root"/patches/*.py "$app/patches/"
cp "$root/web/index.html" "$app/web/"
cp -r "$root/web/assets/"* "$app/web/assets/"
cat > "$bin/b70-launcher" <<EOF
#!/bin/sh
exec python3 "$app/launcher.py" "\$@"
EOF
chmod 755 "$bin/b70-launcher"
icon="$app/web/assets/b70-launcher-256.png"
[ -f "$icon" ] || icon="$app/web/assets/b70-launcher.svg"
cat > "$desktop/b70-launcher.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=B70 Launcher
Comment=Inspect local hardware and launch selected B70 inference recipes
Exec=$bin/b70-launcher
Icon=$icon
Terminal=false
Categories=Utility;
StartupWMClass=b70-launcher
EOF
chmod 644 "$desktop/b70-launcher.desktop"
printf 'Installed %s and %s\nNo engine, model, Docker image, or power setting changed.\n' "$bin/b70-launcher" "$desktop/b70-launcher.desktop"
