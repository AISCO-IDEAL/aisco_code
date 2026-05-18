/**
 * Shared Three.js mesh viewer — ES module.
 */

import { PerspectiveCamera, Scene, Color, WebGLRenderer, HemisphereLight, DirectionalLight, AmbientLight, MeshBasicMaterial, MeshStandardMaterial, MeshPhongMaterial, Mesh, Vector3, Box3, SRGBColorSpace, DoubleSide } from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
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

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    let onControlsChange = null;
    let suppressControlsCallback = false;

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
        };
    }

    function setViewState(viewState) {
        if (!viewState || !viewState.camera || !viewState.target) {
            return;
        }
        suppressControlsCallback = true;
        camera.position.set(viewState.camera.x, viewState.camera.y, viewState.camera.z);
        controls.target.set(viewState.target.x, viewState.target.y, viewState.target.z);
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
        controls.update();
    }

    function fitGlbCamera(root) {
        const box = new Box3().setFromObject(root);
        const center = box.getCenter(new Vector3());
        const size = box.getSize(new Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        controls.target.copy(center);
        camera.up.set(0, 0, 1);
        camera.position.copy(center).addScaledVector(new Vector3(0.4, 1, 0.4).normalize(), Math.max(maxDim, 1e-3) * 2.5);
        camera.near = Math.max(maxDim * 0.001, 0.001);
        camera.far = Math.max(maxDim * 100, 1000);
        camera.updateProjectionMatrix();
        controls.update();
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
                        scene.add(currentObj);
                        fitGlbCamera(currentObj);

                        if (onLoad) {
                            onLoad({
                                hasVertexColors,
                                attributes: hasVertexColors ? ["color"] : [],
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
                const hasVertexColors = geo.hasAttribute("color");
                const material = hasVertexColors
                    ? new MeshPhongMaterial({ vertexColors: true, side: DoubleSide, shininess: 30 })
                    : new MeshStandardMaterial({ color: 0x99A3A4, roughness: 0.6, metalness: 0.2, side: DoubleSide });
                currentObj = new Mesh(geo, material);
                currentObj.scale.set(scale, scale, scale);
                scene.add(currentObj);
                frameScaledObject(size.clone().multiplyScalar(scale));
                if (onLoad) {
                    onLoad({
                        hasVertexColors,
                        attributes: Object.keys(geo.attributes || {}),
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

    return { loadMesh, getViewState, setViewState, setOnControlsChange };
}
