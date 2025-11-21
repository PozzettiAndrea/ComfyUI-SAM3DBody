/**
 * Copyright (c) 2025 Andrea Pozzetti
 * SPDX-License-Identifier: MIT
 *
 * ComfyUI SAM3DBody - Skeleton Preview Widget
 * Interactive 3D skeleton viewer using Three.js
 */

import { app } from "../../../../scripts/app.js";

console.log("[SAM3DBody] Loading skeleton viewer extension...");

// Inline minimal Three.js viewer HTML for skeleton only
const SKELETON_VIEWER_HTML = `
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            background: #2a2a2a;
        }
        #canvas-container {
            width: 100%;
            height: 100vh;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: #ffffff;
            font-family: monospace;
            font-size: 14px;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div id="canvas-container"></div>
    <div id="info">Loading skeleton...</div>

    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    }
    </script>

    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

        console.log('[SAM3DBody Skeleton Viewer] Initializing...');

        let scene, camera, renderer, controls;
        let skeletonGroup;

        function init() {
            const container = document.getElementById('canvas-container');

            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x2a2a2a);

            // Camera
            camera = new THREE.PerspectiveCamera(
                50,
                container.clientWidth / container.clientHeight,
                0.1,
                1000
            );
            camera.position.set(2, 1, 3);
            camera.lookAt(0, 0, 0);

            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            // Controls
            controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 10, 7.5);
            scene.add(directionalLight);

            // Grid helper
            const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x333333);
            scene.add(gridHelper);

            // Axes helper
            const axesHelper = new THREE.AxesHelper(1);
            scene.add(axesHelper);

            // Skeleton group
            skeletonGroup = new THREE.Group();
            scene.add(skeletonGroup);

            // Handle window resize
            window.addEventListener('resize', onWindowResize);

            // Animation loop
            animate();

            console.log('[SAM3DBody Skeleton Viewer] Ready');

            // Notify parent that viewer is ready
            window.parent.postMessage({ type: 'SKELETON_VIEWER_READY' }, '*');
        }

        function onWindowResize() {
            const container = document.getElementById('canvas-container');
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        function loadSkeleton(skeletonData) {
            console.log('[SAM3DBody Skeleton Viewer] Loading skeleton data...', skeletonData);

            // Clear existing skeleton
            while (skeletonGroup.children.length > 0) {
                skeletonGroup.remove(skeletonGroup.children[0]);
            }

            if (!skeletonData || !skeletonData.joint_positions) {
                console.error('[SAM3DBody Skeleton Viewer] No joint positions in skeleton data');
                return;
            }

            const joints = skeletonData.joint_positions;
            const numJoints = joints.length;

            console.log(\`[SAM3DBody Skeleton Viewer] Rendering \${numJoints} joints\`);

            // Find bounding box for auto-scaling
            let minX = Infinity, maxX = -Infinity;
            let minY = Infinity, maxY = -Infinity;
            let minZ = Infinity, maxZ = -Infinity;

            for (const joint of joints) {
                minX = Math.min(minX, joint[0]);
                maxX = Math.max(maxX, joint[0]);
                minY = Math.min(minY, joint[1]);
                maxY = Math.max(maxY, joint[1]);
                minZ = Math.min(minZ, joint[2]);
                maxZ = Math.max(maxZ, joint[2]);
            }

            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            const centerZ = (minZ + maxZ) / 2;

            const rangeX = maxX - minX;
            const rangeY = maxY - minY;
            const rangeZ = maxZ - minZ;
            const maxRange = Math.max(rangeX, rangeY, rangeZ);

            // Normalize scale (target size ~2 units)
            const targetSize = 2.0;
            const scale = maxRange > 0 ? targetSize / maxRange : 1.0;

            console.log(\`[SAM3DBody Skeleton Viewer] Skeleton bounds: [\${minX.toFixed(2)}, \${maxX.toFixed(2)}], [\${minY.toFixed(2)}, \${maxY.toFixed(2)}], [\${minZ.toFixed(2)}, \${maxZ.toFixed(2)}]\`);
            console.log(\`[SAM3DBody Skeleton Viewer] Center: [\${centerX.toFixed(2)}, \${centerY.toFixed(2)}, \${centerZ.toFixed(2)}], Scale: \${scale.toFixed(3)}\`);

            // Create materials
            const jointMaterial = new THREE.MeshPhongMaterial({
                color: 0x00ff88,
                emissive: 0x00ff88,
                emissiveIntensity: 0.3
            });

            const boneMaterial = new THREE.MeshPhongMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 0.6
            });

            const jointGeometry = new THREE.SphereGeometry(0.015, 16, 16);

            // Create joints
            const jointMeshes = [];
            for (let i = 0; i < numJoints; i++) {
                const joint = joints[i];
                const mesh = new THREE.Mesh(jointGeometry, jointMaterial);

                // Apply normalization
                mesh.position.set(
                    (joint[0] - centerX) * scale,
                    (joint[1] - centerY) * scale,
                    (joint[2] - centerZ) * scale
                );

                skeletonGroup.add(mesh);
                jointMeshes.push(mesh);
            }

            // Create bones between nearby joints
            const connectionThreshold = 0.4 * scale; // Adjust threshold based on scale
            const bones = [];

            for (let i = 0; i < numJoints; i++) {
                for (let j = i + 1; j < numJoints; j++) {
                    const pos1 = jointMeshes[i].position;
                    const pos2 = jointMeshes[j].position;

                    const distance = pos1.distanceTo(pos2);

                    if (distance < connectionThreshold && distance > 0.001) {
                        // Create cylinder bone
                        const direction = new THREE.Vector3().subVectors(pos2, pos1);
                        const length = direction.length();

                        const boneGeometry = new THREE.CylinderGeometry(0.005, 0.005, length, 8);
                        const boneMesh = new THREE.Mesh(boneGeometry, boneMaterial);

                        // Position and orient cylinder
                        boneMesh.position.copy(pos1).add(direction.clone().multiplyScalar(0.5));
                        boneMesh.quaternion.setFromUnitVectors(
                            new THREE.Vector3(0, 1, 0),
                            direction.normalize()
                        );

                        skeletonGroup.add(boneMesh);
                        bones.push(boneMesh);
                    }
                }
            }

            console.log(\`[SAM3DBody Skeleton Viewer] Created \${bones.length} bone connections\`);

            // Update info
            document.getElementById('info').textContent =
                \`Skeleton: \${numJoints} joints, \${bones.length} bones\\n\` +
                \`Left click + drag: Rotate\\n\` +
                \`Scroll: Zoom\\n\` +
                \`Right click + drag: Pan\`;

            // Center camera on skeleton
            controls.target.set(0, 0, 0);
            controls.update();
        }

        // Listen for messages from parent
        window.addEventListener('message', (event) => {
            console.log('[SAM3DBody Skeleton Viewer] Received message:', event.data);

            if (event.data && event.data.type === 'LOAD_SKELETON') {
                const filepath = event.data.filepath;
                console.log(\`[SAM3DBody Skeleton Viewer] Loading skeleton from: \${filepath}\`);

                fetch(filepath)
                    .then(response => response.json())
                    .then(data => {
                        console.log('[SAM3DBody Skeleton Viewer] Skeleton data received');
                        loadSkeleton(data);
                    })
                    .catch(error => {
                        console.error('[SAM3DBody Skeleton Viewer] Error loading skeleton:', error);
                        document.getElementById('info').textContent = 'Error loading skeleton';
                    });
            }
        });

        // Initialize viewer
        init();
    </script>
</body>
</html>
`;

app.registerExtension({
    name: "sam3dbody.skeletonviewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SAM3DBodyPreviewSkeleton") {
            console.log("[SAM3DBody] Registering Preview Skeleton node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log("[SAM3DBody] Node created, adding skeleton viewer widget");

                // Create iframe for skeleton viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";

                // Create blob URL from inline HTML
                const blob = new Blob([SKELETON_VIEWER_HTML], { type: 'text/html' });
                const blobUrl = URL.createObjectURL(blob);
                iframe.src = blobUrl;
                console.log('[SAM3DBody] Setting iframe src to blob URL (fully self-contained)');

                // Clean up blob URL when iframe is removed
                iframe.addEventListener('load', () => {
                    iframe._blobUrl = blobUrl;
                });

                // Add load event listener
                iframe.onload = () => {
                    console.log('[SAM3DBody] Skeleton viewer iframe loaded successfully');
                };
                iframe.onerror = (e) => {
                    console.error('[SAM3DBody] Skeleton viewer iframe failed to load:', e);
                };

                // Add widget
                const widget = this.addDOMWidget("skeleton_preview", "SKELETON_VIEWER", iframe, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                console.log("[SAM3DBody] Widget created:", widget);

                // Set widget size
                widget.computeSize = function(width) {
                    const w = width || 512;
                    const h = w; // Square aspect ratio
                    return [w, h];
                };

                widget.element = iframe;

                // Store iframe reference
                this.skeletonViewerIframe = iframe;
                this.skeletonViewerReady = false;

                // Listen for ready message from iframe
                const onMessage = (event) => {
                    if (event.data && event.data.type === 'SKELETON_VIEWER_READY') {
                        console.log('[SAM3DBody] Skeleton viewer iframe is ready!');
                        this.skeletonViewerReady = true;
                    }
                };
                window.addEventListener('message', onMessage.bind(this));

                // Set initial node size
                this.setSize([512, 512]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[SAM3DBody] onExecuted called with message:", message);
                    onExecuted?.apply(this, arguments);

                    // The message contains the skeleton file path
                    if (message?.skeleton_file && message.skeleton_file[0]) {
                        const filename = message.skeleton_file[0];
                        console.log(`[SAM3DBody] Loading skeleton: ${filename}`);

                        // Build filepath
                        const filepath = `${window.location.origin}/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;
                        console.log(`[SAM3DBody] Skeleton path: ${filepath}`);

                        // Send message to iframe (wait for ready or use delay)
                        const sendMessage = () => {
                            if (iframe.contentWindow) {
                                console.log(`[SAM3DBody] Sending postMessage to iframe: ${filepath}`);
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_SKELETON",
                                    filepath: filepath,
                                    timestamp: Date.now()
                                }, "*");
                            } else {
                                console.error("[SAM3DBody] Iframe contentWindow not available");
                            }
                        };

                        // Wait for iframe to be ready, or use timeout as fallback
                        if (this.skeletonViewerReady) {
                            sendMessage();
                        } else {
                            const checkReady = setInterval(() => {
                                if (this.skeletonViewerReady) {
                                    clearInterval(checkReady);
                                    sendMessage();
                                }
                            }, 50);

                            // Fallback timeout after 2 seconds
                            setTimeout(() => {
                                clearInterval(checkReady);
                                if (!this.skeletonViewerReady) {
                                    console.warn("[SAM3DBody] Iframe not ready after 2s, sending anyway");
                                    sendMessage();
                                }
                            }, 2000);
                        }
                    } else {
                        console.log("[SAM3DBody] No skeleton_file in message data. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[SAM3DBody] Skeleton viewer extension registered");
