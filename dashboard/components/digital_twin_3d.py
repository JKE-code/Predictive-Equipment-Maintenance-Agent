"""
3D Digital Twin Component for Predictive Equipment Maintenance Agent.
Renders an interactive WebGL Three.js milling spindle with real-time rotation,
thermal shader glow, torque vibration oscillations, and subsystem fault highlights.
"""

def generate_digital_twin_html(
    speed_rpm: float = 1500.0,
    temp_k: float = 308.0,
    torque_nm: float = 40.0,
    anomaly_score: float = 15.0,
    failure_mode: str = "NORMAL",
    is_failed: bool = False,
    height: int = 420,
) -> str:
    """Generate self-contained Three.js HTML canvas representing the 3D machine spindle."""

    # Map physical temperature (300K - 315K) to 0.0 - 1.0 thermal factor
    temp_factor = max(0.0, min(1.0, (temp_k - 304.0) / 10.0))
    # Map speed to rotation angular velocity
    rotation_speed = max(0.02, min(0.35, (speed_rpm / 1500.0) * 0.12))
    # Map anomaly score to vibration jitter amplitude
    vibration_amplitude = max(0.0, min(0.06, (anomaly_score / 100.0) * 0.05))

    mode_clean = failure_mode.upper()
    hdf_highlight = "true" if "HDF" in mode_clean or "HEAT" in mode_clean else "false"
    pwf_highlight = "true" if "PWF" in mode_clean or "POWER" in mode_clean else "false"
    osf_highlight = "true" if "OSF" in mode_clean or "STRAIN" in mode_clean or "TWF" in mode_clean else "false"
    tripped_highlight = "true" if is_failed or "CRITICAL" in mode_clean else "false"

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body, html {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                background-color: #0b0f19;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            #canvas-container {{
                width: 100%;
                height: {height}px;
                position: relative;
            }}
            .hud-overlay {{
                position: absolute;
                top: 12px;
                left: 14px;
                background: rgba(15, 23, 42, 0.75);
                border: 1px solid rgba(56, 189, 248, 0.25);
                backdrop-filter: blur(8px);
                border-radius: 8px;
                padding: 8px 14px;
                color: #e2e8f0;
                font-size: 11px;
                pointer-events: none;
                z-index: 10;
                line-height: 1.5;
            }}
            .hud-title {{
                font-weight: 700;
                color: #38bdf8;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 2px;
            }}
            .hud-badge {{
                display: inline-block;
                padding: 2px 6px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 10px;
                margin-top: 4px;
            }}
            .badge-normal {{ background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }}
            .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }}
            .badge-critical {{ background: rgba(239, 68, 68, 0.25); color: #ef4444; border: 1px solid #ef4444; animation: pulse 1s infinite; }}
            @keyframes pulse {{
                0% {{ opacity: 0.6; }}
                50% {{ opacity: 1; }}
                100% {{ opacity: 0.6; }}
            }}
            .controls-hint {{
                position: absolute;
                bottom: 8px;
                right: 12px;
                color: #64748b;
                font-size: 10px;
                pointer-events: none;
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <div id="canvas-container">
            <div class="hud-overlay">
                <div class="hud-title">3D Machine Digital Twin</div>
                <div>Spindle Speed: <b>{speed_rpm:.0f} RPM</b></div>
                <div>Process Core: <b>{temp_k:.1f} K</b> ({temp_k - 273.15:.1f}°C)</div>
                <div>Mechanical Torque: <b>{torque_nm:.1f} Nm</b></div>
                <div>Vibration Jitter: <b>{vibration_amplitude*100:.1f} mm/s</b></div>
                <div class="hud-badge {'badge-critical' if is_failed or 'CRITICAL' in mode_clean else ('badge-warning' if 'WARNING' in mode_clean or 'WATCH' in mode_clean else 'badge-normal')}">
                    {mode_clean if mode_clean != 'NORMAL' else 'NOMINAL STATUS'}
                </div>
            </div>
            <div class="controls-hint">🖱️ Left-Click & Drag to Rotate | Scroll to Zoom</div>
        </div>

        <script>
            const container = document.getElementById('canvas-container');
            const width = container.clientWidth || window.innerWidth;
            const height = {height};

            // Scene, Camera, Renderer
            const scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x0b0f19, 0.025);

            const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
            camera.position.set(4.5, 3.2, 5.0);

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(width, height);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.appendChild(renderer.domElement);

            // Lighting
            const ambientLight = new THREE.AmbientLight(0x334155, 1.2);
            scene.add(ambientLight);

            const keyLight = new THREE.DirectionalLight(0xe2e8f0, 1.5);
            keyLight.position.set(5, 8, 4);
            keyLight.castShadow = true;
            scene.add(keyLight);

            const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.8);
            rimLight.position.set(-5, 3, -3);
            scene.add(rimLight);

            // Point Light for Thermal Glow at Spindle Tip
            const tempFactor = {temp_factor};
            const thermalColor = new THREE.Color();
            if (tempFactor > 0.6) {{
                thermalColor.setRGB(1.0, 0.2 + (1.0 - tempFactor) * 0.4, 0.1);
            }} else if (tempFactor > 0.3) {{
                thermalColor.setRGB(0.95, 0.65, 0.2);
            }} else {{
                thermalColor.setRGB(0.22, 0.74, 0.97);
            }}

            const thermalLight = new THREE.PointLight(thermalColor, 1.5 + tempFactor * 2.5, 4.0);
            thermalLight.position.set(0, 0.2, 0);
            scene.add(thermalLight);

            // Grid Platform
            const grid = new THREE.GridHelper(10, 20, 0x1e293b, 0x0f172a);
            grid.position.y = -1.6;
            scene.add(grid);

            // Digital Twin Model Hierarchy
            const machineGroup = new THREE.Group();
            scene.add(machineGroup);

            // 1. Heavy Machine Base
            const baseGeo = new THREE.BoxGeometry(2.6, 0.4, 2.6);
            const baseMat = new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.7, metalness: 0.5 }});
            const baseMesh = new THREE.Mesh(baseGeo, baseMat);
            baseMesh.position.y = -1.4;
            baseMesh.receiveShadow = true;
            machineGroup.add(baseMesh);

            // 2. Vertical Gantry Column
            const colGeo = new THREE.BoxGeometry(0.8, 2.8, 0.8);
            const colMat = new THREE.MeshStandardMaterial({{ color: 0x334155, roughness: 0.5, metalness: 0.6 }});
            const colMesh = new THREE.Mesh(colGeo, colMat);
            colMesh.position.set(0, 0.0, -0.8);
            colMesh.castShadow = true;
            machineGroup.add(colMesh);

            // 3. Spindle Carriage Mount
            const mountGeo = new THREE.BoxGeometry(1.2, 0.6, 1.1);
            const mountMat = new THREE.MeshStandardMaterial({{
                color: {pwf_highlight} ? 0xef4444 : 0x475569,
                roughness: 0.4,
                metalness: 0.7,
                emissive: {pwf_highlight} ? 0x991b1b : 0x000000,
                emissiveIntensity: {pwf_highlight} ? 0.6 : 0.0
            }});
            const mountMesh = new THREE.Mesh(mountGeo, mountMat);
            mountMesh.position.set(0, 0.6, -0.2);
            machineGroup.add(mountMesh);

            // 4. Spindle Housing (Motor Enclosure)
            const housingGeo = new THREE.CylinderGeometry(0.42, 0.42, 1.4, 32);
            const housingMat = new THREE.MeshStandardMaterial({{
                color: {hdf_highlight} ? 0xf97316 : 0x64748b,
                roughness: 0.3,
                metalness: 0.8,
                emissive: {hdf_highlight} ? 0xc2410c : 0x000000,
                emissiveIntensity: {hdf_highlight} ? 0.7 : 0.0
            }});
            const housingMesh = new THREE.Mesh(housingGeo, housingMat);
            housingMesh.position.set(0, 0.5, 0.1);
            housingMesh.castShadow = true;
            machineGroup.add(housingMesh);

            // Cooling Fin Rings on Spindle Housing
            for (let i = 0; i < 4; i++) {{
                const finGeo = new THREE.TorusGeometry(0.48, 0.03, 16, 32);
                const finMat = new THREE.MeshStandardMaterial({{
                    color: {hdf_highlight} ? 0xef4444 : 0x94a3b8,
                    roughness: 0.2,
                    metalness: 0.9,
                    emissive: {hdf_highlight} ? 0xdc2626 : 0x000000,
                    emissiveIntensity: {hdf_highlight} ? 0.8 : 0.0
                }});
                const finMesh = new THREE.Mesh(finGeo, finMat);
                finMesh.rotation.x = Math.PI / 2;
                finMesh.position.set(0, 0.8 - i * 0.18, 0.1);
                machineGroup.add(finMesh);
            }}

            // 5. Rotating Spindle Shaft & Cutter Group
            const rotatingGroup = new THREE.Group();
            rotatingGroup.position.set(0, -0.2, 0.1);
            machineGroup.add(rotatingGroup);

            // Tool Chuck
            const chuckGeo = new THREE.CylinderGeometry(0.3, 0.22, 0.45, 24);
            const chuckMat = new THREE.MeshStandardMaterial({{ color: 0x0f172a, roughness: 0.3, metalness: 0.9 }});
            const chuckMesh = new THREE.Mesh(chuckGeo, chuckMat);
            chuckMesh.position.y = -0.15;
            rotatingGroup.add(chuckMesh);

            // Cutting Tool (End Mill with Thermal Shader Glow)
            const toolGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.65, 16);
            const toolMat = new THREE.MeshStandardMaterial({{
                color: thermalColor,
                roughness: 0.2,
                metalness: 0.85,
                emissive: thermalColor,
                emissiveIntensity: 0.3 + tempFactor * 1.2
            }});
            const toolMesh = new THREE.Mesh(toolGeo, toolMat);
            toolMesh.position.y = -0.65;
            toolMesh.castShadow = true;
            rotatingGroup.add(toolMesh);

            // Fluted Tool Tip (Cutting Zone)
            const tipGeo = new THREE.ConeGeometry(0.09, 0.25, 16);
            const tipMat = new THREE.MeshStandardMaterial({{
                color: {osf_highlight} ? 0xef4444 : thermalColor,
                roughness: 0.1,
                metalness: 0.95,
                emissive: {osf_highlight} ? 0xef4444 : thermalColor,
                emissiveIntensity: {osf_highlight} ? 1.5 : (0.4 + tempFactor * 1.5)
            }});
            const tipMesh = new THREE.Mesh(tipGeo, tipMat);
            tipMesh.rotation.x = Math.PI;
            tipMesh.position.y = -1.05;
            rotatingGroup.add(tipMesh);

            // Warning Holographic Ring (Active if critical or tripped)
            const warningRingGeo = new THREE.RingGeometry(0.9, 1.05, 32);
            const warningRingMat = new THREE.MeshBasicMaterial({{
                color: {tripped_highlight} ? 0xef4444 : 0x38bdf8,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: {tripped_highlight} ? 0.8 : 0.25
            }});
            const warningRing = new THREE.Mesh(warningRingGeo, warningRingMat);
            warningRing.rotation.x = Math.PI / 2;
            warningRing.position.set(0, -1.35, 0.1);
            machineGroup.add(warningRing);

            // Camera Mouse Interaction (Smooth Orbit Controls)
            let isDragging = false;
            let prevMouseX = 0;
            let prevMouseY = 0;
            let targetRotY = 0.5;
            let targetRotX = 0.2;
            let cameraDist = 6.2;

            container.addEventListener('mousedown', (e) => {{
                isDragging = true;
                prevMouseX = e.clientX;
                prevMouseY = e.clientY;
            }});

            window.addEventListener('mouseup', () => {{
                isDragging = false;
            }});

            window.addEventListener('mousemove', (e) => {{
                if (!isDragging) return;
                const deltaX = e.clientX - prevMouseX;
                const deltaY = e.clientY - prevMouseY;
                targetRotY += deltaX * 0.008;
                targetRotX = Math.max(-0.2, Math.min(0.8, targetRotX + deltaY * 0.008));
                prevMouseX = e.clientX;
                prevMouseY = e.clientY;
            }});

            container.addEventListener('wheel', (e) => {{
                cameraDist = Math.max(3.5, Math.min(9.0, cameraDist + e.deltaY * 0.005));
                e.preventDefault();
            }});

            // Animation Loop
            let clock = new THREE.Clock();
            const rotationSpeed = {rotation_speed};
            const vibrationAmp = {vibration_amplitude};
            const isTripped = {tripped_highlight};

            function animate() {{
                requestAnimationFrame(animate);
                const elapsedTime = clock.getElapsedTime();

                // 1. Spindle Tool Rotation (if not tripped)
                if (!isTripped) {{
                    rotatingGroup.rotation.y += rotationSpeed;
                }}

                // 2. Torque / Anomaly Mechanical Jitter (Vibration)
                if (vibrationAmp > 0.005) {{
                    const jitterX = (Math.random() - 0.5) * vibrationAmp;
                    const jitterZ = (Math.random() - 0.5) * vibrationAmp;
                    housingMesh.position.x = jitterX;
                    housingMesh.position.z = 0.1 + jitterZ;
                    rotatingGroup.position.x = jitterX;
                    rotatingGroup.position.z = 0.1 + jitterZ;
                }} else {{
                    housingMesh.position.x = 0;
                    housingMesh.position.z = 0.1;
                    rotatingGroup.position.x = 0;
                    rotatingGroup.position.z = 0.1;
                }}

                // 3. Pulsing Alert Ring
                if (isTripped) {{
                    warningRing.scale.setScalar(1.0 + Math.sin(elapsedTime * 6.0) * 0.15);
                    warningRingMat.opacity = 0.5 + Math.sin(elapsedTime * 6.0) * 0.4;
                }} else {{
                    warningRing.rotation.z += 0.01;
                }}

                // 4. Camera Position from Orbital Angles
                camera.position.x = Math.sin(targetRotY) * Math.cos(targetRotX) * cameraDist;
                camera.position.y = Math.sin(targetRotX) * cameraDist + 0.8;
                camera.position.z = Math.cos(targetRotY) * Math.cos(targetRotX) * cameraDist;
                camera.lookAt(0, -0.2, 0);

                renderer.render(scene, camera);
            }}

            animate();

            // Resize Handler
            window.addEventListener('resize', () => {{
                const newW = container.clientWidth || window.innerWidth;
                camera.aspect = newW / height;
                camera.updateProjectionMatrix();
                renderer.setSize(newW, height);
            }});
        </script>
    </body>
    </html>
    """
    return html
