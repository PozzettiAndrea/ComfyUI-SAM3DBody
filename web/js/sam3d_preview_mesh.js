/**
 * Copyright (c) 2025 Andrea Pozzetti
 * SPDX-License-Identifier: MIT
 *
 * ComfyUI SAM3DBody - Rigged Mesh Preview Widget
 * Interactive viewer for SAM3D rigged meshes with skeleton manipulation
 *
 * Uses file-based viewer loading from comfy-3d-viewers package.
 * The viewer_fbx.html and Three.js bundle are copied by prestartup_script.py.
 */

import { app } from "../../../../scripts/app.js";

console.log("[SAM3DBody] Loading rigged mesh preview extension...");

/**
 * Detect the extension folder name from the current script URL
 */
function detectExtensionFolder() {
    try {
        if (typeof import.meta !== 'undefined' && import.meta.url) {
            const match = import.meta.url.match(/\/extensions\/([^\/]+)\//);
            if (match) {
                return match[1];
            }
        }
    } catch (e) {
        console.warn('[SAM3DBody] Could not detect extension folder from import.meta:', e);
    }
    // Fallback to hardcoded name
    return 'ComfyUI-SAM3DBody';
}

/**
 * Get the viewer URL for the FBX viewer HTML file
 */
function getViewerUrl(extensionFolder) {
    return `/extensions/${extensionFolder}/viewer_fbx.html?v=` + Date.now();
}

const EXTENSION_FOLDER = detectExtensionFolder();
console.log("[SAM3DBody] Detected extension folder:", EXTENSION_FOLDER);

app.registerExtension({
    name: "sam3dbody.meshpreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SAM3DBodyPreviewRiggedMesh") {
            console.log("[SAM3DBody] Registering Preview Rigged Mesh node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log("[SAM3DBody] Node created, adding FBX viewer widget");

                // Create iframe for FBX viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";

                // Load viewer from file URL (copied by prestartup_script.py)
                const viewerUrl = getViewerUrl(EXTENSION_FOLDER);
                iframe.src = viewerUrl;
                console.log('[SAM3DBody] Setting iframe src to:', viewerUrl);

                // Add load event listener
                iframe.onload = () => {
                    console.log('[SAM3DBody] Iframe loaded successfully');
                };
                iframe.onerror = (e) => {
                    console.error('[SAM3DBody] Iframe failed to load:', e);
                };

                // Add widget
                const widget = this.addDOMWidget("preview", "FBX_PREVIEW", iframe, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                console.log("[SAM3DBody] Widget created:", widget);

                // Set widget size - allow flexible height
                widget.computeSize = function(width) {
                    const w = width || 512;
                    const h = w * 1.5;  // Taller than wide to accommodate controls
                    return [w, h];
                };

                widget.element = iframe;

                // Store iframe reference
                this.fbxViewerIframe = iframe;
                this.fbxViewerReady = false;

                // Listen for ready message from iframe
                const onMessage = (event) => {
                    if (event.data && event.data.type === 'VIEWER_READY') {
                        console.log('[SAM3DBody] Viewer iframe is ready!');
                        this.fbxViewerReady = true;
                    }
                };
                window.addEventListener('message', onMessage.bind(this));

                const notifyIframeResize = () => {
                    if (iframe.contentWindow) {
                        const rect = iframe.getBoundingClientRect();
                        iframe.contentWindow.postMessage({
                            type: 'RESIZE',
                            width: rect.width,
                            height: rect.height
                        }, '*');
                    }
                };

                this.onResize = function(size) {
                    const isVueNodes = iframe.closest('[data-node-id]') !== null ||
                                       document.querySelector('.vue-graph-canvas') !== null;

                    if (!isVueNodes && size && size[1]) {
                        const nodeHeight = size[1];
                        const headerHeight = 70;
                        const availableHeight = Math.max(200, nodeHeight - headerHeight);
                        iframe.style.height = availableHeight + 'px';
                    }

                    requestAnimationFrame(() => {
                        notifyIframeResize();
                    });
                };

                let resizeTimeout = null;
                let lastSize = { width: 0, height: 0 };
                const resizeObserver = new ResizeObserver((entries) => {
                    const entry = entries[0];
                    const newWidth = entry.contentRect.width;
                    const newHeight = entry.contentRect.height;

                    if (Math.abs(newWidth - lastSize.width) < 1 && Math.abs(newHeight - lastSize.height) < 1) {
                        return;
                    }
                    lastSize = { width: newWidth, height: newHeight };

                    if (resizeTimeout) {
                        clearTimeout(resizeTimeout);
                    }
                    resizeTimeout = setTimeout(() => {
                        notifyIframeResize();
                    }, 50);
                });
                resizeObserver.observe(iframe);

                const originalOnRemoved = this.onRemoved;
                this.onRemoved = function() {
                    resizeObserver.disconnect();
                    if (resizeTimeout) {
                        clearTimeout(resizeTimeout);
                    }
                    window.removeEventListener('message', onMessage);
                    if (originalOnRemoved) {
                        originalOnRemoved.apply(this, arguments);
                    }
                };

                // Set initial node size (taller to accommodate controls)
                this.setSize([512, 768]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[SAM3DBody] onExecuted called with message:", message);
                    onExecuted?.apply(this, arguments);

                    // The message contains the FBX file path
                    if (message?.fbx_file && message.fbx_file[0]) {
                        const filename = message.fbx_file[0];
                        console.log(`[SAM3DBody] Loading FBX: ${filename}`);

                        // Try different path formats based on filename
                        let filepath;

                        // If filename is just a basename, it's in output
                        if (!filename.includes('/') && !filename.includes('\\')) {
                            // Try output directory first - use absolute URL for blob iframe
                            filepath = `${window.location.origin}/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;
                            console.log(`[SAM3DBody] Using output path: ${filepath}`);
                        } else {
                            // Full path - extract just the filename
                            const basename = filename.split(/[/\\]/).pop();
                            filepath = `${window.location.origin}/view?filename=${encodeURIComponent(basename)}&type=output&subfolder=`;
                            console.log(`[SAM3DBody] Extracted basename: ${basename}, path: ${filepath}`);
                        }

                        // Send message to iframe (wait for ready or use delay)
                        const sendMessage = () => {
                            if (iframe.contentWindow) {
                                console.log(`[SAM3DBody] Sending postMessage to iframe: ${filepath}`);
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_FBX",
                                    filepath: filepath,
                                    timestamp: Date.now()
                                    // Note: SAM3DBody doesn't have FBX export API, so no fbxExportApiPath
                                }, "*");
                            } else {
                                console.error("[SAM3DBody] Iframe contentWindow not available");
                            }
                        };

                        // Wait for iframe to be ready, or use timeout as fallback
                        if (this.fbxViewerReady) {
                            sendMessage();
                        } else {
                            const checkReady = setInterval(() => {
                                if (this.fbxViewerReady) {
                                    clearInterval(checkReady);
                                    sendMessage();
                                }
                            }, 50);

                            // Fallback timeout after 2 seconds
                            setTimeout(() => {
                                clearInterval(checkReady);
                                if (!this.fbxViewerReady) {
                                    console.warn("[SAM3DBody] Iframe not ready after 2s, sending anyway");
                                    sendMessage();
                                }
                            }, 2000);
                        }
                    } else {
                        console.log("[SAM3DBody] No fbx_file in message data. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[SAM3DBody] Rigged mesh preview extension registered");
