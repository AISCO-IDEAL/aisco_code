/**
 * Shared Three.js mesh viewer — ES module.
 *
 * Loaded per-template via a dynamic import() so it participates in the
 * existing importmap that resolves "three" from the unpkg CDN:
 *
 *   import("/static/js/mesh_viewer.js").then(({ initMeshViewer }) => { ... });
 *
 * The importmap must already be present on the page (defined in the
 * template's {% block head %} before this module is imported).
 */

import { PerspectiveCamera, Scene, Color, WebGLRenderer, HemisphereLight, DirectionalLight, MeshStandardMaterial, Mesh, Vector3 } from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

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
    renderer.setSize(containerEl.clientWidth, containerEl.clientHeight);
    containerEl.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new HemisphereLight(0xffffff, 0x444444, 1.0));
    const dl = new DirectionalLight(0xffffff, 0.8);
    dl.position.set(1, 1, 1);
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

    /**
     * Load (or replace) the PLY mesh at url.
     * @param {string} url - URL to the .ply file (cache-busted by caller)
     * @param {function=} onLoad - optional callback invoked after the mesh is added to the scene
     */
    function loadMesh(url, onLoad) {
        new PLYLoader().load(
            url,
            (geo) => {
                geo.computeVertexNormals();
                geo.computeBoundingBox();
                const c = new Vector3();
                geo.boundingBox.getCenter(c);
                geo.translate(-c.x, -c.y, -c.z);
                geo.computeBoundingBox();
                const sz = new Vector3();
                geo.boundingBox.getSize(sz);
                const scale = Math.max(sz.x, sz.y, sz.z) > 0 ? 2.0 / Math.max(sz.x, sz.y, sz.z) : 1;
                if (currentObj) {
                    scene.remove(currentObj);
                    currentObj.geometry.dispose();
                    currentObj.material.dispose();
                }
                currentObj = new Mesh(geo, new MeshStandardMaterial({ color: 0x99A3A4, roughness: 0.6, metalness: 0.2, side: 2 }));
                currentObj.scale.set(scale, scale, scale);
                scene.add(currentObj);
                camera.position.set(0, 0, 3.5);
                controls.target.set(0, 0, 0);
                controls.update();
                if (onLoad) onLoad();
            },
            null,
            (err) => console.error("PLY load error:", err),
        );
    }

    return { loadMesh };
}
