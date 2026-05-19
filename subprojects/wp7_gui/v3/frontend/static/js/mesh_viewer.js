/**
 * Shared Three.js mesh viewer — ES module.
 */

import { PerspectiveCamera, Scene, Color, WebGLRenderer, HemisphereLight, DirectionalLight, AmbientLight, MeshBasicMaterial, MeshStandardMaterial, MeshPhongMaterial, Mesh, Vector3, Box3, SRGBColorSpace, DoubleSide } from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

let glbConfigPromise = null;

function loadGlbConfig() {
    if (!glbConfigPromise) {
        glbConfigPromise = fetch("/api/artifacts/glb-config", { cache: "no-store" })
            .then((resp) => (resp.ok ? resp.json() : {}))
            .catch(() => ({}));
    }
    return glbConfigPromise;
}

function buildGlbColorMap(config) {
    return {
        sdf_mesh: {
            rgb: Array.isArray(config?.sdf_color) ? config.sdf_color : null,
            alpha: typeof config?.sdf_alpha === "number" ? config.sdf_alpha : null,
        },
        inner_assembly: {
            rgb: Array.isArray(config?.inner_color) ? config.inner_color : null,
            alpha: null,
        },
        outer_bbox: {
            rgb: Array.isArray(config?.outer_color) ? config.outer_color : null,
            alpha: typeof config?.outer_alpha === "number" ? config.outer_alpha : null,
        },
    };
}

/**
 * Initialise a Three.js PLY viewer inside containerEl and return a load
 * function that replaces the displayed mesh on each call.
 *
 * @param {HTMLElement} containerEl - the element that will host the renderer
 * @returns {{ loadMesh: function(string): void }}
 */
export function initMeshViewer(containerEl) {
    const scene = new Scene();
    scene.background = new Color(0xf1f7f3);

    const camera = new PerspectiveCamera(50, containerEl.clientWidth / containerEl.clientHeight, 0.1, 1000);
    camera.position.set(0, 0, 3);

    const renderer = new WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(containerEl.clientWidth, containerEl.clientHeight);
    renderer.outputColorSpace = SRGBColorSpace;
    containerEl.appendChild(renderer.domElement);

    const controls = new TrackballControls(camera, renderer.domElement);
    let onControlsChange = null;
    let suppressControlsCallback = false;
    let glbControlsHost = null;

    function ensureOpacitySliderStyles() {
        const styleId = "glb-opacity-slider-styles";
        if (document.getElementById(styleId)) {
            return;
        }
        const style = document.createElement("style");
        style.id = styleId;
        style.textContent = `
            .glb-opacity-slider {
                -webkit-appearance: auto;
                appearance: auto;
                height: 14px;
                background: transparent;
                padding: 0;
                margin: 0;
                outline: none;
                accent-color: #c33531;
            }
        `;
        document.head.appendChild(style);
    }

    function paintOpacitySliderBar(slider) {
        if (!slider) {
            return;
        }
        slider.style.accentColor = "#c33531";
    }

    scene.add(new HemisphereLight(0xffffff, 0x444444, 1.0));
    scene.add(new AmbientLight(0xffffff, 0.6));
    const dl = new DirectionalLight(0xffffff, 0.8);
    dl.position.set(5, 10, 7);
    scene.add(dl);

    window.addEventListener("resize", () => {
        camera.aspect = containerEl.clientWidth / containerEl.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(containerEl.clientWidth, containerEl.clientHeight);
    });

    let currentObj = null;
    let currentGlbComponents = [];

    (function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    })();

    controls.addEventListener("change", () => {
        if (suppressControlsCallback) {
            return;
        }
        if (typeof onControlsChange === "function") {
            onControlsChange(getViewState());
        }
    });

    function getViewState() {
        return {
            camera: {
                x: camera.position.x,
                y: camera.position.y,
                z: camera.position.z,
            },
            target: {
                x: controls.target.x,
                y: controls.target.y,
                z: controls.target.z,
            },
            up: {
                x: camera.up.x,
                y: camera.up.y,
                z: camera.up.z,
            },
            quaternion: {
                x: camera.quaternion.x,
                y: camera.quaternion.y,
                z: camera.quaternion.z,
                w: camera.quaternion.w,
            },
        };
    }

    function setViewState(viewState) {
        if (!viewState || !viewState.camera || !viewState.target) {
            return;
        }
        suppressControlsCallback = true;
        camera.position.set(viewState.camera.x, viewState.camera.y, viewState.camera.z);
        controls.target.set(viewState.target.x, viewState.target.y, viewState.target.z);
        if (viewState.up) {
            camera.up.set(viewState.up.x, viewState.up.y, viewState.up.z);
        }
        if (viewState.quaternion) {
            camera.quaternion.set(
                viewState.quaternion.x,
                viewState.quaternion.y,
                viewState.quaternion.z,
                viewState.quaternion.w,
            );
        }
        controls.update();
        suppressControlsCallback = false;
    }

    function setOnControlsChange(callback) {
        onControlsChange = callback;
    }

    function disposeObject(obj) {
        obj.traverse((node) => {
            if (!node.isMesh) {
                return;
            }
            if (node.geometry) {
                node.geometry.dispose();
            }
            const mats = Array.isArray(node.material) ? node.material : [node.material];
            mats.forEach((m) => {
                if (m && typeof m.dispose === "function") {
                    m.dispose();
                }
            });
        });
    }

    function clearGlbComponents() {
        currentGlbComponents = [];
    }

    function getMeshComponents() {
        return currentGlbComponents.map((entry) => {
            const mats = Array.isArray(entry.mesh.material) ? entry.mesh.material : [entry.mesh.material];
            const firstOpacity = mats.length && typeof mats[0]?.opacity === "number" ? mats[0].opacity : 1.0;
            return {
                key: entry.key,
                name: entry.name,
                visible: entry.mesh.visible,
                opacity: firstOpacity,
            };
        });
    }

    function setMeshComponentState(key, nextState) {
        const entry = currentGlbComponents.find((item) => item.key === key);
        if (!entry || !nextState) {
            return;
        }
        if (Object.prototype.hasOwnProperty.call(nextState, "visible")) {
            entry.mesh.visible = !!nextState.visible;
        }
        if (Object.prototype.hasOwnProperty.call(nextState, "opacity")) {
            const opacity = Math.min(1, Math.max(0, Number(nextState.opacity)));
            const mats = Array.isArray(entry.mesh.material) ? entry.mesh.material : [entry.mesh.material];
            mats.forEach((m) => {
                if (!m) {
                    return;
                }
                m.opacity = opacity;
                m.transparent = true;
                m.depthWrite = opacity >= 0.99;
                m.needsUpdate = true;
            });
        }
    }

    function renderGlbControls() {
        if (!glbControlsHost) {
            return;
        }
        ensureOpacitySliderStyles();
        glbControlsHost.innerHTML = "";
        const components = getMeshComponents();
        if (!components.length) {
            glbControlsHost.style.display = "none";
            return;
        }
        glbControlsHost.style.display = "grid";
        glbControlsHost.style.gridTemplateColumns = "repeat(3, minmax(0, 1fr))";
        glbControlsHost.style.gap = "8px";

        const order = ["inner_assembly", "sdf_mesh", "outer_bbox"];
        const sorted = components.slice().sort((a, b) => {
            const ai = order.indexOf(a.name);
            const bi = order.indexOf(b.name);
            const aRank = ai === -1 ? 99 : ai;
            const bRank = bi === -1 ? 99 : bi;
            return aRank - bRank;
        });

        sorted.forEach((component) => {
            const cell = document.createElement("div");
            cell.style.display = "grid";
            cell.style.gridTemplateColumns = "auto 1fr";
            cell.style.columnGap = "6px";
            cell.style.rowGap = "4px";
            cell.style.alignItems = "center";

            const visible = document.createElement("input");
            visible.type = "checkbox";
            visible.checked = !!component.visible;
            visible.title = "Visible";

            const name = document.createElement("span");
            name.textContent = component.name.replace(/_/g, " ");
            name.style.fontSize = "0.68rem";
            name.style.color = "#000000";
            name.style.whiteSpace = "nowrap";
            name.style.overflow = "hidden";
            name.style.textOverflow = "ellipsis";

            const opacity = document.createElement("input");
            opacity.type = "range";
            opacity.min = "0";
            opacity.max = "1";
            opacity.step = "0.01";
            opacity.value = String(component.opacity);
            opacity.className = "glb-opacity-slider";
            opacity.style.gridColumn = "1 / span 2";
            opacity.style.width = "100%";
            paintOpacitySliderBar(opacity);

            visible.addEventListener("change", () => {
                setMeshComponentState(component.key, { visible: visible.checked });
            });
            opacity.addEventListener("input", () => {
                setMeshComponentState(component.key, { opacity: parseFloat(opacity.value) });
                paintOpacitySliderBar(opacity);
            });

            cell.appendChild(visible);
            cell.appendChild(name);
            cell.appendChild(opacity);
            glbControlsHost.appendChild(cell);
        });
    }

    function setGlbControlsHost(hostEl) {
        glbControlsHost = hostEl || null;
        renderGlbControls();
    }

    function frameScaledObject(sizeVec) {
        const vFovHalf = 25 * Math.PI / 180; // half of the 50° FOV
        const aspect = containerEl.clientWidth / Math.max(containerEl.clientHeight, 1);
        const hFovHalf = Math.atan(Math.tan(vFovHalf) * aspect);
        const dFitW = (sizeVec.x / 2) / Math.tan(hFovHalf);
        const dFitH = (sizeVec.y / 2) / Math.tan(vFovHalf);
        const radius = Math.max(sizeVec.x, sizeVec.y, sizeVec.z) / 2;
        const dFitSphere = radius > 0 ? radius / Math.sin(vFovHalf) : 2.5;
        camera.position.set(0, 0, Math.max(dFitW, dFitH, dFitSphere) * 1.2);
        camera.near = 0.01;
        camera.far = 5000;
        camera.updateProjectionMatrix();
        controls.target.set(0, 0, 0);
        camera.quaternion.identity();
        controls.update();
        // Force TrackballControls to sync its internal state with the camera
        const currentState = getViewState();
        setViewState(currentState);
    }

    function fitGlbCamera(root) {
        const box = new Box3().setFromObject(root);
        const center = box.getCenter(new Vector3());
        const size = box.getSize(new Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);

        // Normalize to the same ~2-unit bounding box used by the PLY path so that
        // camera sync works correctly across file types (shared camera positions).
        const scale = maxDim > 0 ? 2.0 / maxDim : 1;
        root.scale.setScalar(scale);
        root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);

        camera.quaternion.identity();
        frameScaledObject(size.clone().multiplyScalar(scale));
    }

    function getUrlPath(url) {
        // Artifact URLs have the form /api/artifacts/file?path=...filename.glb
        // The bare pathname (/api/artifacts/file) has no extension, so we must
        // check the 'path' query parameter first for the real file extension.
        try {
            const u = new URL(url, window.location.origin);
            const pathParam = u.searchParams.get("path");
            if (pathParam) {
                return pathParam.split("?")[0].split("#")[0].toLowerCase();
            }
        } catch (_) {
            // Non-URL strings fall through to the simple split below.
        }
        return String(url || "").split("?")[0].split("#")[0].toLowerCase();
    }

    /**
     * Load (or replace) the mesh at url.
     * Supports .ply and .glb files.
     * @param {string} url - URL to the mesh file (cache-busted by caller)
    * @param {function=} onLoad - optional callback invoked after the mesh is added to the scene
    * @param {function=} onError - optional callback invoked when mesh load fails
     */
    function loadMesh(url, onLoad, onError) {
        const cleanPath = getUrlPath(url);
        if (cleanPath.endsWith(".glb")) {
            loadGlbConfig().then((config) => {
                const meshColors = buildGlbColorMap(config);
                new GLTFLoader().load(
                    url,
                    (gltf) => {
                        const root = gltf.scene;
                        if (!root) {
                            if (typeof onError === "function") {
                                onError(new Error("GLB load error: missing scene root"));
                            }
                            return;
                        }

                        let hasVertexColors = false;
                        let meshCount = 0;
                        const glbComponents = [];
                        root.traverse((node) => {
                            if (!node.isMesh || !node.geometry) {
                                return;
                            }
                            meshCount += 1;
                            const cfgEntry = meshColors[node.name] || { rgb: null, alpha: null };
                            const srcMat = Array.isArray(node.material) ? node.material[0] : node.material;
                            const baseAlpha = cfgEntry.alpha ?? (typeof srcMat?.opacity === "number" ? srcMat.opacity : 1.0);
                            const useVertexColors = cfgEntry.rgb === null && node.geometry.hasAttribute("color");
                            if (useVertexColors) {
                                hasVertexColors = true;
                            }
                            node.material = new MeshBasicMaterial({
                                vertexColors: useVertexColors,
                                color: cfgEntry.rgb ? new Color(...cfgEntry.rgb) : 0xffffff,
                                side: DoubleSide,
                                transparent: true,
                                opacity: baseAlpha,
                                depthWrite: baseAlpha >= 0.99,
                            });
                            glbComponents.push({
                                key: node.uuid,
                                name: node.name || `mesh_${node.id}`,
                                mesh: node,
                            });
                        });

                        if (meshCount === 0) {
                            if (typeof onError === "function") {
                                onError(new Error("GLB scene has no mesh nodes"));
                            }
                            return;
                        }

                        if (currentObj) {
                            scene.remove(currentObj);
                            disposeObject(currentObj);
                        }
                        currentObj = root;
                        currentGlbComponents = glbComponents;
                        scene.add(currentObj);
                        fitGlbCamera(currentObj);
                        renderGlbControls();

                        if (onLoad) {
                            onLoad({
                                hasVertexColors,
                                attributes: hasVertexColors ? ["color"] : [],
                                meshComponents: getMeshComponents(),
                            });
                        }
                    },
                    undefined,
                    (err) => {
                        if (typeof onError === "function") {
                            onError(err);
                        }
                    },
                );
            });
            return;
        }

        new PLYLoader().load(
            url,
            (geo) => {
                geo.computeVertexNormals();
                geo.computeBoundingBox();
                const center = new Vector3();
                geo.boundingBox.getCenter(center);
                geo.translate(-center.x, -center.y, -center.z);
                geo.computeBoundingBox();
                const size = new Vector3();
                geo.boundingBox.getSize(size);
                const scale = Math.max(size.x, size.y, size.z) > 0 ? 2.0 / Math.max(size.x, size.y, size.z) : 1;
                if (currentObj) {
                    scene.remove(currentObj);
                    disposeObject(currentObj);
                }
                clearGlbComponents();
                const hasVertexColors = geo.hasAttribute("color");
                const material = hasVertexColors
                    ? new MeshPhongMaterial({ vertexColors: true, side: DoubleSide, shininess: 30 })
                    : new MeshStandardMaterial({ color: 0x99A3A4, roughness: 0.6, metalness: 0.2, side: DoubleSide });
                currentObj = new Mesh(geo, material);
                currentObj.scale.set(scale, scale, scale);
                scene.add(currentObj);
                frameScaledObject(size.clone().multiplyScalar(scale));
                renderGlbControls();
                if (onLoad) {
                    onLoad({
                        hasVertexColors,
                        attributes: Object.keys(geo.attributes || {}),
                        meshComponents: [],
                    });
                }
            },
            null,
            (err) => {
                if (typeof onError === "function") {
                    onError(err);
                }
            },
        );
    }

    function refreshGlbControls() {
        renderGlbControls();
    }

    return { loadMesh, getViewState, setViewState, setOnControlsChange, setGlbControlsHost, getMeshComponents, setMeshComponentState, refreshGlbControls };
}
