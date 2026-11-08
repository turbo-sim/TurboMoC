"""Opt-in real-browser check of the control-point workspace.

Run with the app dependencies and Playwright available. Uses a small synthetic
MoC wall so the test exercises the real app without a thermodynamic solve.
Artifacts are written to output/control_point_ux; no user cases are modified.
"""
from pathlib import Path
import json
import logging
import os
import runpy
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from turbo_moc.app import create_app, workflow as flow
from turbo_moc.configuration import project as projects
from turbo_moc.geometry.stator_blade import parametrize_stator
from turbo_moc.geometry.passage_2d import blade_contour
from werkzeug.serving import make_server
import numpy as np
import yaml

# The local prototype may use the sibling checkout's browser-test dependency.
sys.path.append(str(ROOT.parent / "nurbspy/.venv/Lib/site-packages"))
from playwright.sync_api import sync_playwright

OUT = ROOT / ("output/control_point_ux_mirrored" if os.environ.get("TURBO_MOC_TEST_MIRROR") == "1"
              else "output/control_point_ux")
OUT.mkdir(parents=True, exist_ok=True)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
nozzle, config = runpy.run_path(str(ROOT / "tests/conftest.py"))["small_design"].__wrapped__()
config["blade_geometry"].update(n_cp=30, num_baseline_points=500, num_curve_points=500)
config["meridional_channel"]["num_blades"] = 31
config["blade_geometry"]["control_point_displacements"] = {10: [0.001, -0.002]}
config["blade_geometry"]["mirror"] = os.environ.get("TURBO_MOC_TEST_MIRROR") == "1"
case = OUT / "case.yaml"
case.write_text(yaml.safe_dump(projects.normalize_project(config)), encoding="utf-8")
app = create_app(case, output_dir=OUT / "artifacts")
registry = {"stator_1": {"nozzle": {"session": "ux-check", "job": "moc"}}}
flow.load_nozzle = lambda *args: nozzle
flow.resolved_configuration = lambda project, row, refs: projects.row_configuration(project, row)
server = make_server("127.0.0.1", 0, app.server, threaded=True)
threading.Thread(target=server.serve_forever, daemon=True).start()


def idle(page):
    return page.wait_for_function("""() => {
        const v=window.testViewer;
        if (!v || v.disposed || v.pending || v.inFlight || v.drag || v.resampling ||
            v.editing.box || v.editing.history.locked || v.displayQuality!=='full') return false;
        return !document.getElementById('cp-apply').disabled ? {model:structuredClone(v.acceptedModel)} : false;
    }""").json_value()


def frame(page):
    return page.evaluate("""() => {
        const v=window.testViewer, r=v.renderer.domElement.getBoundingClientRect();
        return {camera:[...v.camera.position.toArray(),...v.camera.quaternion.toArray(),
            ...v.controls.target.toArray(),v.camera.zoom], rectangle:[r.x,r.y,r.width,r.height]};
    }""")


def assert_stable(page, expected):
    actual = frame(page)
    assert np.allclose(actual["camera"], expected["camera"], rtol=0, atol=1e-12), (expected, actual)
    assert np.allclose(actual["rectangle"], expected["rectangle"], rtol=0, atol=0.1), (expected, actual)


def handle_position(page, index):
    return page.evaluate("""i => {
        const v=window.testViewer, r=v.renderer.domElement.getBoundingClientRect(), f=v.plotRectangle(r.width,r.height);
        const p=v.handles[i].position.clone().project(v.camera);
        return {x:r.x+f.left+(p.x+1)*f.width/2,y:r.y+f.top+(1-p.y)*f.height/2};
    }""", index)


def seed_references(page):
    # Model an already computed case, preserving its user-selected blade count
    # instead of triggering the app's fresh-MoC automatic sizing action.
    page.evaluate("""refs => {
        window.dash_clientside.set_props('applied-blade-count-jobs', {data:{stator_1:['ux-check','moc']}});
        window.dash_clientside.set_props('row-references', {data:refs});
    }""", registry)


def travel(page, redo=False):
    page.get_by_label("Redo geometry edit" if redo else "Undo geometry edit", exact=True).click()
    return idle(page)


errors, http_errors, rejected_edits, checks = [], [], [], []
try:
    with sync_playwright() as p:
        executable = os.environ.get("TURBO_MOC_TEST_CHROMIUM")
        if not executable:
            candidates = list((Path(os.environ["LOCALAPPDATA"]) / "ms-playwright").glob("chromium-*/chrome-win64/chrome.exe"))
            executable = str(sorted(candidates)[-1]) if candidates else None
        browser = p.chromium.launch(executable_path=executable, headless=True, args=["--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        page.set_default_timeout(15000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        def record_response(response):
            if response.status == 422 and response.url.endswith('/_turbomoc_cp/evaluate'):
                rejected_edits.append(response)
            elif response.status >= 400:
                http_errors.append([response.status, response.url])
        page.on("response", record_response)
        page.goto(f"http://127.0.0.1:{server.server_port}", wait_until="networkidle")
        # Instrument the native camera only in the test, without a production debug API.
        page.evaluate("""async () => {
            const {Explorer}=await import('/_turbomoc_nurbspy/explorer.js');
            const fit=Explorer.prototype.fit;
            Explorer.prototype.fit=function(...args){window.testViewer=this;return fit.apply(this,args);};
        }""")
        seed_references(page)
        page.wait_for_function("!document.getElementById('cp-edit-open').disabled")
        page.locator("#cp-edit-open").click()
        original = idle(page)
        page.locator("#cp-editor").scroll_into_view_if_needed()
        page.screenshot(path=str(OUT / "editor-desktop.png"))
        assert page.get_by_label("Selected control point").is_visible()
        assert page.get_by_label("X coordinate").is_visible()
        assert page.get_by_role("button", name="Reset point", exact=True).is_visible()
        for label in ("Quantity", "Show frames", "Control point weight", "Z coordinate", "Projection", "Geometry colour"):
            assert page.get_by_label(label, exact=True).count() == 0 or not page.get_by_label(label, exact=True).first.is_visible(), label
        assert page.locator("#cp-viewport .nrb-sidebar:visible").count() == 1
        assert page.evaluate("window.testViewer.handles.length") == 30
        assert page.evaluate("window.testViewer.displayPreset") == "fine"
        assert page.evaluate("""JSON.stringify(window.testViewer.topologies.preview) ===
            JSON.stringify(window.testViewer.topologies.full)""")
        assert page.evaluate("window.testViewer.adjacentCurves.length") == 2
        assert page.evaluate("window.testViewer.adjacentCurves.every(l=>l.geometry===window.testViewer.geometryObject.geometry)")
        wall = nozzle["wall_final"]
        native = parametrize_stator(wall["x"], wall["y"], **config["blade_geometry"])
        baseline = parametrize_stator(wall["x"], wall["y"], **{
            key: value for key, value in config["blade_geometry"].items()
            if key not in ("control_point_offsets", "control_point_displacements")})
        baseline_points = np.pad(np.asarray(baseline['control_points']).T, ((0,1),(0,0)))
        shifts = page.evaluate("window.testViewer.adjacentCurves.map(line=>line.position.x)")
        with app.server.app_context():
            design = flow.get_design(config, registry["stator_1"]["nozzle"])
        pitch = design.blade["pitch"] * np.ptp(blade_contour(native)[:, 1]) / np.ptp(blade_contour(design.blade)[:, 1])
        assert np.allclose(shifts, [-pitch, pitch], rtol=0, atol=1e-10), (shifts, pitch)
        checks.extend(["one focused pane", "periodic live neighbours without handles"])

        page.get_by_label("Selected control point").select_option("10")
        before = frame(page)
        untouched = idle(page)["model"]
        page.get_by_label("X coordinate").press("Enter")
        assert idle(page)["model"] == untouched
        old_x = float(page.get_by_label("X coordinate").input_value())
        page.get_by_label("X coordinate").fill(str(old_x + 0.05))
        page.get_by_label("X coordinate").press("Enter")
        edited = idle(page)
        assert abs(edited["model"]["control_points"][0][10] - old_x - 0.05) < 1e-12
        assert_stable(page, before)
        page.get_by_role("button", name="Reset point", exact=True).click()
        page.wait_for_function("window.testViewer.localGeneration > 1")
        idle(page)
        assert_stable(page, before)
        assert abs(float(page.get_by_label("X displacement [mm]", exact=True).text_content())) < 1e-12
        checks.append("numeric edit and Reset point preserve camera and canvas")
        reset_point = idle(page)
        assert travel(page)["model"] == edited["model"]
        assert travel(page, redo=True)["model"] == reset_point["model"]
        assert_stable(page, before)
        checks.append("Reset point is undoable and redoable")

        # Check layout and camera on every pointer movement, including pending previews.
        position = handle_position(page, 10)
        base_point = page.evaluate("window.testViewer.handles[10].position.toArray()")
        page.mouse.move(position["x"], position["y"])
        page.mouse.down()
        assert page.evaluate("Boolean(window.testViewer.drag)")
        for i in range(1, 7):
            page.mouse.move(position["x"] + i * 2, position["y"] - i * 1.5)
            assert_stable(page, before)
        page.mouse.up()
        dragged = idle(page)
        assert_stable(page, before)
        moved = handle_position(page, 10)
        assert abs(moved["x"] - position["x"] - 12) < 0.3
        assert abs(moved["y"] - position["y"] + 9) < 0.3
        assert dragged["model"]["control_points"][0][10] != base_point[0]
        assert dragged["model"]["control_points"][1][10] != base_point[1]
        assert page.evaluate("window.testViewer.adjacentCurves.every(l=>l.geometry===window.testViewer.geometryObject.geometry)")
        checks.append("pointer drag follows cursor without camera or layout drift")
        page.screenshot(path=str(OUT / "editor-dragged.png"))

        # Resets participate in the same native history as drags and numeric edits.
        generation = page.evaluate('window.testViewer.localGeneration')
        page.locator('#cp-reset').click()
        page.wait_for_function('generation => window.testViewer.localGeneration > generation', arg=generation)
        reset_all = idle(page)
        assert np.allclose(reset_all['model']['control_points'], baseline_points, rtol=0, atol=1e-13), np.max(
            np.abs(np.asarray(reset_all['model']['control_points']) - baseline_points))
        assert travel(page)["model"] == dragged["model"]
        assert travel(page, redo=True)["model"] == reset_all["model"]
        assert travel(page)["model"] == dragged["model"]
        assert_stable(page, before)
        checks.append("Reset all is undoable and redoable")

        # Reuse native Shift-click and group dragging; fixed handles cannot join.
        page.get_by_label('Selected control point').select_option('10')
        position = handle_position(page, 12)
        page.keyboard.down('Shift')
        page.mouse.click(position['x'], position['y'])
        page.keyboard.up('Shift')
        assert page.evaluate('[...window.testViewer.selectedIndices].sort((a,b)=>a-b)') == [10, 12]
        page.mouse.move(position['x'], position['y'])
        page.mouse.down()
        assert page.evaluate('window.testViewer.drag.starts.map(([i])=>i).sort((a,b)=>a-b)') == [10, 12]
        for step in range(1, 5):
            page.mouse.move(position['x'] + step, position['y'] - step / 2)
            assert_stable(page, before)
        page.mouse.up()
        grouped = idle(page)
        delta = np.asarray(grouped['model']['control_points']) - np.asarray(dragged['model']['control_points'])
        assert np.allclose(delta[:,10], delta[:,12], rtol=0, atol=1e-13)
        assert np.any(delta[:,10])
        assert np.all(delta[:,[i for i in range(30) if i not in [10,12]]] == 0)
        assert travel(page)['model'] == dragged['model']
        assert travel(page, redo=True)['model'] == grouped['model']
        assert travel(page)['model'] == dragged['model']

        page.get_by_role('button', name='Box select', exact=True).click()
        rectangle = page.evaluate("""() => {
            const v=window.testViewer,r=v.renderer.domElement.getBoundingClientRect(),p=v.plotRectangle(r.width,r.height);
            return {x:r.x+p.left,y:r.y+p.top,width:p.width,height:p.height};
        }""")
        page.mouse.move(rectangle['x']+2, rectangle['y']+2)
        page.mouse.down()
        page.mouse.move(rectangle['x']+rectangle['width']-2, rectangle['y']+rectangle['height']-2, steps=5)
        page.mouse.up()
        assert page.evaluate('[...window.testViewer.selectedIndices].sort((a,b)=>a-b)') == list(range(2,28))
        assert idle(page)['model'] == dragged['model']
        page.mouse.click(rectangle['x']+5, rectangle['y']+5)
        assert page.evaluate('window.testViewer.selectedIndices.size') == 0
        page.evaluate('window.testViewer.editing.updateHistory()')
        assert page.get_by_label('X coordinate').is_disabled()
        assert page.get_by_role('button', name='Reset point', exact=True).is_disabled()
        assert page.get_by_label('X displacement [mm]', exact=True).text_content() == '—'
        for index in [0,1,28,29]:
            assert page.locator(f'.cp-point-panel option[value="{index}"]').get_attribute('disabled') is not None
        page.get_by_label('Selected control point').select_option('10')
        checks.append("Shift/box selection and group undo preserve fixed points and camera")

        # Escape must cancel a drag even after its preview has reached Python.
        history_size = page.evaluate('window.testViewer.editing.history.past.length')
        position = handle_position(page, 10)
        page.mouse.move(position['x'], position['y'])
        page.mouse.down()
        page.mouse.move(position['x']+6, position['y']-2)
        page.wait_for_function("""() => window.testViewer.drag && !window.testViewer.pending &&
            !window.testViewer.inFlight && window.testViewer.displayQuality === 'preview'""")
        assert page.evaluate("""Number(window.testViewer.root.dataset.sampleCount) ===
            window.testViewer.topologies.full.count""")
        assert page.evaluate("window.testViewer.adjacentCurves.every(l=>l.geometry===window.testViewer.geometryObject.geometry)")
        page.keyboard.press('Escape')
        page.mouse.up()
        assert idle(page)['model'] == dragged['model']
        assert page.evaluate('window.testViewer.editing.history.past.length') == history_size
        assert page.get_by_label('X coordinate').is_visible()
        assert_stable(page, before)
        checks.append("fine sampling stays active during drag; Escape restores the pre-drag shape")

        # Invalid edits roll back through native history and never become undo steps.
        history_size = page.evaluate('window.testViewer.editing.history.past.length')
        page.get_by_label("X coordinate").fill("-10000" if config['blade_geometry']['mirror'] else "10000")
        page.get_by_label("X coordinate").press("Enter")
        page.wait_for_function("document.getElementById('cp-message').classList.contains('cp-error')")
        assert idle(page)['model'] == dragged['model']
        assert page.evaluate('window.testViewer.editing.history.past.length') == history_size
        assert 'intersects' in page.locator('#cp-message').text_content()
        assert len(rejected_edits) == 1
        assert page.locator("#cp-reset").is_enabled()
        assert_stable(page, before)
        checks.append("invalid edit rolls back without entering history; valid blade remains applicable")

        for label in ("Adjacent blades", "Original outline", "Control polygon", "Reference grid"):
            page.get_by_label(label, exact=True).uncheck()
            assert_stable(page, before)
            page.get_by_label(label, exact=True).check()
            assert_stable(page, before)
        page.locator(".cp-view-options > summary").click()
        assert_stable(page, before)
        page.locator(".cp-view-options > summary").click()
        page.keyboard.press("Escape")
        assert page.get_by_label("X coordinate").is_visible()
        assert_stable(page, before)
        assert_stable(page, before)
        checks.append("view switches and selection do not move the drawing")

        page.set_viewport_size({"width": 1200, "height": 900})
        page.wait_for_function("window.testViewer.renderer.domElement.clientWidth < 800")
        page.get_by_role("button", name="Fit passage", exact=True).click()
        page.screenshot(path=str(OUT / "editor-compact.png"))
        assert page.get_by_label("X coordinate").is_visible()
        assert page.locator("#cp-viewport .nrb-canvas").bounding_box()["width"] > 350
        assert page.locator(".cp-footer").bounding_box()["y"] + 36 < 855
        last_switch = page.get_by_label("Adjacent blades", exact=True).bounding_box()
        assert last_switch["y"] + last_switch["height"] < page.locator(".cp-footer").bounding_box()["y"] - 5
        checks.append("compact desktop layout")
        page.locator("#cp-apply").click()
        page.wait_for_function("document.getElementById('cp-editor').style.display==='none'")
        with page.expect_download() as info:
            page.locator("#save-configuration").click()
        saved = OUT / "saved.yaml"
        info.value.save_as(saved)
        page.locator("#configuration-upload input[type=file]").set_input_files(saved)
        page.wait_for_function("document.getElementById('configuration-message').textContent===''")
        page.wait_for_timeout(250)
        seed_references(page)
        page.wait_for_function("!document.getElementById('cp-edit-open').disabled")
        page.locator("#cp-edit-open").click()
        reopened = idle(page)
        assert np.allclose(reopened["model"]["control_points"], dragged["model"]["control_points"], rtol=0, atol=1e-13)
        page.locator("#cp-cancel").click()
        page.wait_for_function("document.getElementById('cp-editor').style.display==='none'")
        checks.append("Apply, YAML download/upload, reconstruction and Cancel")
        assert not errors, errors
        assert not http_errors, http_errors
        result = {"result": "passed", "checks": checks, "page_errors": errors, "http_errors": http_errors}
        (OUT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        browser.close()
finally:
    server.shutdown()
