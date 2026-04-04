#!/bin/bash
set -e

APP_NAME="VideoSubtitleExtractor"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_PATH="${PROJECT_DIR}/dist/${APP_NAME}.app"
DMG_NAME="${APP_NAME}"
DMG_PATH="${PROJECT_DIR}/dist/${DMG_NAME}.dmg"
VOLUME_NAME="${APP_NAME}"

BG_IMAGE="${PROJECT_DIR}/design/bg.png"
ICON_PATH="${PROJECT_DIR}/build/icon.icns"

# Sanity checks
if [ ! -d "${APP_PATH}" ]; then
    echo "Error: ${APP_PATH} not found. Run PyInstaller first."
    exit 1
fi
if [ ! -f "${BG_IMAGE}" ]; then
    echo "Error: Background image ${BG_IMAGE} not found."
    exit 1
fi

# Remove old DMG if exists
rm -f "${DMG_PATH}"

# Create a temporary DMG staging area
STAGING_DIR=$(mktemp -d)
echo "Staging dir: ${STAGING_DIR}"

# Copy app into staging
cp -R "${APP_PATH}" "${STAGING_DIR}/"

# Create a symlink to /Applications for drag-and-drop install
ln -s /Applications "${STAGING_DIR}/Applications"

# --- Create the DMG ---
echo "Creating DMG..."

# Use hdiutil to create a read-write DMG
# Size: app size + buffer for filesystem overhead
APP_SIZE=$(du -sm "${STAGING_DIR}" | cut -f1)
DMG_SIZE=$((APP_SIZE + 200))

hdiutil create -srcfolder "${STAGING_DIR}" \
    -volname "${VOLUME_NAME}" \
    -fs HFS+ \
    -fsargs "-c c=64,a=16,e=16" \
    -format UDRW \
    -size ${DMG_SIZE}m \
    -ov \
    "${DMG_PATH}.rw.dmg"

echo "DMG created. Mounting to set background and icon positions..."

# Mount the DMG
DEVICE=$(hdiutil attach -readwrite -noverify -noautoopen "${DMG_PATH}.rw.dmg" | \
    egrep '^/dev/' | sed 1q | awk '{print $1}')
MOUNT_POINT="/Volumes/${VOLUME_NAME}"
echo "Mounted at: ${MOUNT_POINT} (${DEVICE})"

sleep 2

# Copy background image into the DMG
mkdir -p "${MOUNT_POINT}/.background"
cp "${BG_IMAGE}" "${MOUNT_POINT}/.background/background.png"

# Set the DMG window properties using AppleScript
# Window size should accommodate the background image (600x300)
osascript <<EOF
tell application "Finder"
    tell disk "${VOLUME_NAME}"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {100, 100, 1124, 675}
        set viewOptions to the icon view options of container window
        set arrangement of viewOptions to not arranged
        set icon size of viewOptions to 100
        set background picture of viewOptions to file ".background:background.png"
        set position of item "${APP_NAME}.app" of container window to {330, 310}
        set position of item "Applications" of container window to {730, 310}
        close
        open
        update without registering applications
        delay 5
    end tell
end tell
EOF

echo "Window properties set. Converting to compressed DMG..."

# Make sure everything is flushed
sync
sleep 2

# Detach
hdiutil detach "${DEVICE}" -force
sleep 2

# Convert to compressed read-only DMG
rm -f "${DMG_PATH}"
hdiutil convert "${DMG_PATH}.rw.dmg" \
    -format UDZO \
    -imagekey zlib-level=9 \
    -o "${DMG_PATH}"

# Cleanup
rm -f "${DMG_PATH}.rw.dmg"
rm -rf "${STAGING_DIR}"

echo ""
echo "DMG created successfully: ${DMG_PATH}"
echo "Size: $(du -sh "${DMG_PATH}" | cut -f1)"
