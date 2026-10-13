/* Run with node --test tests/js/test_orientation_cube.js. */
const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");

class Element {
  constructor() { this.children = []; this.style = {}; this.dataset = {}; this.attributes = {}; this.events = {}; }
  setAttribute(key, value) { this.attributes[key] = value; }
  appendChild(child) {
    if (child.parent) child.parent.children = child.parent.children.filter(item => item !== child);
    this.children.push(child); child.parent = this;
  }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  addEventListener(name, callback) { this.events[name] = callback; }
  removeEventListener(name) { delete this.events[name]; }
  focus() { this.focusCount = (this.focusCount || 0) + 1; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this); }
}

function browser(reducedMotion = false) {
  let now = 0, next = 0;
  const frames = new Map();
  const context = vm.createContext({window: {matchMedia: () => ({matches: reducedMotion})},
    document: {createElement: () => new Element(), createElementNS: () => new Element()},
    performance: {now: () => now},
    requestAnimationFrame: callback => { frames.set(++next, callback); return next; },
    cancelAnimationFrame: id => frames.delete(id)});
  vm.runInContext(readFileSync(join(__dirname, "../../src/turbo_moc/app/assets/orientation_cube.js"), "utf8"), context);
  function tick(delta = 50) {
    now += delta;
    const pending = [...frames.values()]; frames.clear();
    pending.forEach(callback => callback(now));
  }
  function flush() { for (let count = 0; frames.size && count < 30; count++) tick(); assert.equal(frames.size, 0); }
  return {cube: context.window.turbomocOrientation, tick, flush, frames, context};
}

function view() {
  let position = [11,22,33], focus = [10,20,30], up = [0,0,1];
  const listeners = new Set();
  const modified = () => listeners.forEach(callback => callback());
  return {listeners, renders: 0, clippingUpdates: 0,
    camera: {
      getPosition: () => position, getFocalPoint: () => focus, getViewUp: () => up,
      setPosition: (...values) => { position = values; modified(); },
      setViewUp: (...values) => { up = values; modified(); },
      // Navigation must never fit bounds, change projection/scale or reset pan.
      setFocalPoint: () => assert.fail("Navigation changed the focal point"),
      setParallelScale: () => assert.fail("Navigation changed parallel zoom"),
      onModified: callback => { listeners.add(callback); return {unsubscribe: () => listeners.delete(callback)}; }
    },
    renderer: {resetCameraClippingRange() {}},
    style: {setCenterOfRotation(value) { assert.deepEqual([...value], focus); }},
    renderWindow: {render() {}}, containerRef: {current: new Element()}};
}

const close = (a, b, tolerance = 1e-10) => {
  assert.equal(a.length, b.length);
  a.forEach((value, i) => assert.ok(Math.abs(value-b[i]) < tolerance, `${a} differs from ${b}`));
};

test("all 26 directions support orthonormal camera frames with both Z-up and Y-up conventions", () => {
  const {cube} = browser();
  for (const up of [[0,0,1], [0,1,0]]) for (const x of [-1,0,1]) for (const y of [-1,0,1]) for (const z of [-1,0,1]) {
    if (!x && !y && !z) continue;
    const target = cube.targetBasis([x,y,z], up);
    const restored = cube.fromQuaternion(cube.quaternion(target));
    close(restored.back, [x,y,z].map(value => value/Math.hypot(x,y,z)));
    close(restored.up, target.up);
    close(restored.right, target.right);
    assert.ok(Math.abs(target.up.reduce((sum, value, i) => sum+value*target.back[i], 0)) < 1e-10);
  }
});

test("the cube has 26 actual polygon targets and shows all eight bevels in a face-on view", () => {
  const {cube, flush} = browser(true);
  const target = view(), widget = cube.create(new Element(), "geometry");
  widget.attach(target); widget.setEnabled(true); widget.snap([-1,0,0], [0,1,0]); flush();
  const nodes = widget.host.children[0].children[1].children;
  assert.equal(nodes.length, 26);
  for (const [kind, count] of [["face",6], ["edge",12], ["corner",8]]) {
    assert.equal(nodes.filter(node => node.attributes["data-kind"] === kind).length, count);
  }
  const visible = nodes.filter(node => node.style.display !== "none");
  assert.equal(visible.length, 9);
  assert.equal(visible.filter(node => node.attributes["data-kind"] === "edge").length, 4);
  assert.equal(visible.filter(node => node.attributes["data-kind"] === "corner").length, 4);
  for (const node of visible) {
    const points = node.children[1].attributes.points.split(" ").map(point => point.split(",").map(Number));
    const area = Math.abs(points.reduce((sum, point, index) => {
      const next = points[(index+1)%points.length];
      return sum + point[0]*next[1]-point[1]*next[0];
    }, 0)/2);
    assert.ok(area > 50, `Bevel target ${node.attributes["data-direction"]} is too small: ${area}`);
    assert.equal(node.attributes.tabindex, "0");
  }
});

test("clicking a visible chamfer selects its edge or corner view", () => {
  const {cube, flush} = browser(true);
  const target = view(), widget = cube.create(new Element(), "mesh");
  widget.attach(target); widget.setEnabled(true); widget.snap([-1,0,0]); flush();
  const nodes = widget.host.children[0].children[1].children;
  for (const direction of [[-1,1,0], [-1,1,1]]) {
    const node = nodes.find(node => node.attributes["data-direction"] === direction.join(","));
    assert.equal(node.style.display, "");
    node.events.click(); flush();
    close(target.camera.getPosition().map((value, i) => value-target.camera.getFocalPoint()[i]),
      direction.map(value => value*Math.sqrt(14)/Math.hypot(...direction)));
  }
});

test("opposite-face animations stay finite and keep an orthonormal camera basis", () => {
  const {cube} = browser();
  const start = cube.quaternion(cube.targetBasis([0,0,1]));
  const end = cube.quaternion(cube.targetBasis([0,0,-1]));
  for (const t of [0, .1, .5, .9, 1]) {
    const frame = cube.fromQuaternion(cube.interpolate(start, end, t));
    assert.ok(Object.values(frame).flat().every(Number.isFinite));
    close(Object.values(frame).map(values => Math.hypot(...values)), [1,1,1]);
  }
});

test("animated navigation preserves camera focus and distance, including both poles", () => {
  const {cube, flush} = browser();
  const target = view(), parent = new Element(), widget = cube.create(parent, "mesh");
  widget.attach(target); widget.setEnabled(true); flush();
  const focus = [...target.camera.getFocalPoint()];
  for (const direction of [[-1,0,0], [0,0,1], [0,0,-1], [1,1,0], [-1,-1,-1]]) {
    widget.snap(direction); flush();
    const offset = target.camera.getPosition().map((value, i) => value-focus[i]);
    close(offset, direction.map(value => value * Math.sqrt(14)/Math.hypot(...direction)));
    close(target.camera.getFocalPoint(), focus);
  }
  assert.equal(widget.host.hidden, false);
});

test("camera subscriptions survive display edits without duplication and are released on detach", () => {
  const {cube, flush, frames} = browser();
  const target = view(), parent = new Element(), widget = cube.create(parent, "geometry");
  widget.attach(target); widget.attach(target); widget.setEnabled(true); flush();
  assert.equal(target.listeners.size, 1);
  const before = widget.host.dataset.orientation;
  target.camera.setPosition(20,20,30); flush();
  assert.notEqual(widget.host.dataset.orientation, before);
  widget.snap([0,1,0]); widget.destroy(); flush();
  assert.equal(target.listeners.size, 0);
  assert.equal(parent.children.length, 0);
  assert.equal(frames.size, 0);
});

test("dragging the main view or hiding its dataset cancels an in-progress rotation", () => {
  const {cube, tick, flush} = browser();
  const target = view(), widget = cube.create(new Element(), "mesh");
  widget.attach(target); widget.setEnabled(true); flush();
  widget.snap([-1,0,0]); tick();
  target.containerRef.current.events.pointerdown();
  const interrupted = [...target.camera.getPosition()];
  flush(); close(target.camera.getPosition(), interrupted);
  widget.snap([0,1,0]); widget.setEnabled(false); flush();
  close(target.camera.getPosition(), interrupted);
  assert.equal(widget.host.hidden, true);
});

test("reduced motion snaps in one frame and each viewer keeps its own camera", () => {
  const {cube, tick, flush} = browser(true);
  const first = view(), second = view();
  const a = cube.create(new Element(), "geometry"), b = cube.create(new Element(), "mesh");
  a.attach(first); b.attach(second); a.setEnabled(true); b.setEnabled(true); flush();
  const untouched = [...second.camera.getPosition()];
  a.snap([0,0,1]); tick();
  close(first.camera.getPosition(), [10,20,30+Math.sqrt(14)]);
  close(second.camera.getPosition(), untouched);
});

test("Home restores the configured engineering basis and does not alter zoom or focus", () => {
  const {cube, flush} = browser(true);
  const target = view(), widget = cube.create(new Element(), "geometry");
  widget.attach(target); widget.setEnabled(true);
  widget.setHome([-1,.2,.15], [0,1,0]);
  widget.snap([0,1,0]); flush();
  widget.host.children[1].children[0].events.click(); flush();
  const expected = cube.targetBasis([-1,.2,.15], [0,1,0]);
  close(target.camera.getPosition(), expected.back.map((value, i) => value*Math.sqrt(14)+[10,20,30][i]));
  close(target.camera.getViewUp(), expected.up);
  widget.snap([0,0,1]); flush();
  close(target.camera.getViewUp(), [0,1,0]);
});

test("orbit and roll controls use fixed angles and preserve the other camera degrees of freedom", () => {
  const {cube, flush} = browser(true);
  const target = view(), widget = cube.create(new Element(), "mesh");
  widget.attach(target); widget.setEnabled(true); widget.snap([0,0,1], [0,1,0]); flush();
  const controls = widget.host.children[0].children[2].children;
  assert.equal(controls.length, 6);
  const click = action => { controls.find(node => node.attributes["data-action"] === action).events.click(); flush(); };
  click("orbit-right");
  close(target.camera.getPosition(), [10+Math.sqrt(14)*.5,20,30+Math.sqrt(14)*Math.sqrt(3)/2]);
  close(target.camera.getViewUp(), [0,1,0]);
  click("orbit-left"); click("orbit-up");
  close(target.camera.getPosition(), [10,20+Math.sqrt(14)*.5,30+Math.sqrt(14)*Math.sqrt(3)/2]);
  click("orbit-down");
  const position = [...target.camera.getPosition()];
  click("roll-right");
  close(target.camera.getPosition(), position);
  close(target.camera.getViewUp(), [-Math.sin(Math.PI/12), Math.cos(Math.PI/12), 0]);
  // The point that was at the top of the image moves right, matching the
  // clockwise arrow and label rather than the opposite camera-roll sign.
  const rolled = cube.basis([0,0,1], target.camera.getViewUp());
  assert.ok(rolled.right[1] > 0);
  assert.ok(rolled.up[1] > 0);
  click("roll-left");
  close(target.camera.getViewUp(), [0,1,0]);
});

test("rapid orbit clicks accumulate full steps while animation is in progress", () => {
  const {cube, tick, flush} = browser();
  const target = view(), widget = cube.create(new Element(), "geometry");
  widget.attach(target); widget.setEnabled(true); widget.snap([0,0,1], [0,1,0]); flush();
  widget.orbit(30,0); tick(); widget.orbit(30,0); tick(); widget.orbit(30,0); flush();
  close(target.camera.getPosition(), [10+Math.sqrt(14),20,30]);
  close(target.camera.getViewUp(), [0,1,0]);
});

test("keyboard navigation transfers focus off faces that may become hidden", () => {
  const {cube, flush} = browser(true);
  const widget = cube.create(new Element(), "mesh");
  widget.attach(view()); widget.setEnabled(true); flush();
  const svg = widget.host.children[0];
  const face = svg.children[1].children.find(node => node.attributes["aria-label"] === "View from +Z");
  face.events.keydown({key: "Enter", preventDefault() {}}); flush();
  assert.equal(svg.focusCount, 1);
  for (let count = 0; count < 3; count++) {
    svg.events.keydown({key: "ArrowRight", preventDefault() {}, stopPropagation() {}}); flush();
  }
  assert.equal(svg.focusCount, 4);
  assert.equal(face.style.display, "none");
});

test("the dash-vtk adapter attaches a standard ref to the lazy VTK class", () => {
  const {cube, context} = browser();
  const suspense = Symbol("Suspense"), lazy = Symbol("View");
  context.React = {Suspense: suspense, isValidElement: value => !!value?.type,
    cloneElement: (element, props, ...children) => ({...element, props: {...element.props, ...props,
      ...(children.length ? {children: children[0]} : {})}})};
  context.window.dash_vtk = {View: props => ({type: suspense, props: {children: {type: lazy, props}}})};
  context.window.dash_vtk.View.defaultProps = {cameraParallelProjection: false};
  const ref = () => {}, child = {type: "test-actor"};
  const element = cube.vtkView({id: "geometry-view"}, [child], ref);
  assert.equal(element.props.children.props.ref, ref);
  assert.equal(element.props.children.props.children[0], child);
  assert.equal(element.props.children.props.cameraParallelProjection, false);
});
