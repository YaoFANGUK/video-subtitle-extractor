#!/bin/bash
 set -e
# PyInstaller post-build: fix libpng conflict
APP_NAME="VideoSubtitleExtractor"
APP_DIR="dist/${APP_NAME}.app"
FRAMEWORKS_DIR="${APP_DIR}/Contents/Frameworks"
RESOURCES_DIR="${APP_DIR}/Contents/Resources"

VSF_LIB_DIR="${RESOURCES_DIR}/backend/subfinder/macos/lib"

FRAMEWORKS_PNG="${FRAMEWORKS_DIR}/libpng16.16.dylib"
if [ -f "${FRAMEWORKS_PNG}" ] && [ -f "${VSF_LIB_DIR}/libpng16.16.dylib" ]; then
    echo "Replacing Frameworks/libpng16 with VSF version (    cp "${VSF_LIB_DIR}/libpng16.16.dylib" "${FRAMEWORKS_PNG}" || \
    for lib in "${VSF_LIB_DIR}/"libopencv_imgproc.4.13.0.dylib" "libopencv_imgcodecs.4.13.0.dylib" \
lib/lib/libavcodec.62.28.100.dylib" \
lib/lib/libavformat.62.12.100.dylib" \
lib/lib/libavutil.60.26.100.dylib" \
lib/lib/libswresample.6.3.100.dylib" \
lib/lib/libswscale.9.5.100.dylib; do
        if [ -f "$lib" ] && [ -f "${VSF_LIB_DIR}/$lib" ]; then
        echo "    Also replacing: $lib"
        cp "$lib" "${FRAMEWORKS_DIR}/" || \
    done
fi

    # Re-sign
 codesign --force --sign - "${FRAMEWORKS_DIR}/"*.dylib" 2>/dev/null
 \
    codesign --force --sign - "${FRAMEWORKS_DIR}/PIL/__dot__dylibs"/*.dylib" 2>/dev/null; \
    codesign --force --sign - "${FRAMEWORKS_DIR}/cv2/__dot__dylibs"/*.dylib" 2>/dev/null; \
    codesign --force --sign - "${FRAMEWORKS_DIR}/backend"*.dylib" 2>/dev/null; \
    codesign --force --sign - "${APP_DIR}" 2>/dev/null; \
    echo "post-build fixes applied"
 || \
    # exec
>> "$BUILD_output"

 || \
    echo "Build_dmg.sh completed successfully"
 || exit "$build_dmg.sh completed with exit code $?"
 >&2
 || exit 1
 || \
    exit 0)
