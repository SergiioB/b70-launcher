#!/bin/sh
# B70 Launcher rootless uninstaller (POSIX sh).
#
# Removes only the files the installer placed under the current user's home:
#   ${XDG_DATA_HOME:-~/.local/share}/b70-launcher
#   ~/.local/bin/b70-launcher and ~/.local/bin/b70
#   ${XDG_DATA_HOME:-~/.local/share}/applications/b70-launcher.desktop
#   ${XDG_DATA_HOME:-~/.local/share}/icons/hicolor/*/apps/b70-launcher.*
#
# Usage history, logs, server state and settings overrides live in
#   ${XDG_STATE_HOME:-~/.local/state}/b70-launcher
# and are KEPT by default. Pass --purge to remove that state as well.
# Model downloads (~/models, ~/Downloads) are never touched.
set -eu

say()  { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [ "$(id -u)" = "0" ]; then
    die "run this as the user who installed B70 Launcher, not root."
fi
[ -n "${HOME:-}" ] || die "HOME is not set; cannot locate ~/.local."

data_home=${XDG_DATA_HOME:-$HOME/.local/share}
state_dir=${XDG_STATE_HOME:-$HOME/.local/state}/b70-launcher
app=$data_home/b70-launcher
wrapper=$HOME/.local/bin/b70-launcher
cli_wrapper=$HOME/.local/bin/b70
desktop_file=$data_home/applications/b70-launcher.desktop

purge=false
for a in "$@"; do
    case $a in
        --purge) purge=true ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) die "unknown option: $a (expected --purge or --help)" ;;
    esac
done

removed=0
note() { removed=$((removed + 1)); say "removed $1"; }

# App directory: guarded wipe.
if [ -e "$app" ]; then
    case $app in
        */b70-launcher) rm -rf -- "$app" && note "$app" ;;
        *) warn "kept unexpected path: $app" ;;
    esac
else
    say "no app directory at $app (already removed or never installed)"
fi

# Wrapper: remove only if it carries our generated marker.
if [ -f "$wrapper" ]; then
    if grep -q 'B70 Launcher wrapper' "$wrapper" 2>/dev/null; then
        rm -f -- "$wrapper" && note "$wrapper"
    else
        warn "kept $wrapper - it does not look like the generated B70 Launcher wrapper; inspect it before deleting."
    fi
fi

# CLI wrapper: same marker check.
if [ -f "$cli_wrapper" ]; then
    if grep -q 'B70 Launcher wrapper' "$cli_wrapper" 2>/dev/null; then
        rm -f -- "$cli_wrapper" && note "$cli_wrapper"
    else
        warn "kept $cli_wrapper - it does not look like the generated B70 Launcher wrapper; inspect it before deleting."
    fi
fi

# Desktop entry: remove only if it references this app.
if [ -f "$desktop_file" ]; then
    if grep -q 'b70-launcher' "$desktop_file" 2>/dev/null; then
        rm -f -- "$desktop_file" && note "$desktop_file"
    else
        warn "kept $desktop_file - it does not reference b70-launcher; inspect it before deleting."
    fi
fi

# Icons: only files literally named b70-launcher.* under hicolor apps dirs.
for f in "$data_home"/icons/hicolor/*/apps/b70-launcher.*; do
    [ -f "$f" ] && rm -f -- "$f" && note "$f"
done

# State directory: kept unless --purge.
if [ -d "$state_dir" ]; then
    if $purge; then
        case $state_dir in
            */b70-launcher) rm -rf -- "$state_dir" && note "$state_dir" ;;
            *) warn "kept unexpected path: $state_dir" ;;
        esac
    else
        say "kept $state_dir - usage history, logs and settings overrides still there."
        say "      remove it with: rm -rf \"$state_dir\"   or re-run: uninstall.sh --purge"
    fi
fi
say "Model downloads under ~/models and ~/Downloads were not touched."

# Best-effort desktop database / icon cache refresh.
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$data_home/applications" >/dev/null 2>&1 || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t "$data_home/icons/hicolor" >/dev/null 2>&1 || true
fi

if [ "$removed" -gt 0 ]; then
    say "B70 Launcher uninstalled ($removed path(s) removed). Nothing outside your home was touched."
else
    say "Nothing to remove - B70 Launcher does not appear to be installed for this user."
fi
