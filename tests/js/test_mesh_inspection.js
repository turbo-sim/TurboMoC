/* Run with node --test tests/js/test_mesh_inspection.js. */
const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");

function browser() {
  const context = vm.createContext({
    window: {dash_clientside: {no_update: "unchanged"}},
    document: {
      addEventListener() {}, querySelectorAll() { return []; },
      getElementById() { return {querySelector() { return {
        width: 400, getBoundingClientRect() { return {width: 200}; }
      }; }}; }
    },
    MutationObserver: class { observe() {} },
    setTimeout() {}, requestAnimationFrame(callback) { callback(); }
  });
  for (const file of ["mesh_edges.js", "studio.js", "inspection.js"])
    vm.runInContext(readFileSync(join(__dirname, "../../src/turbo_moc/app/assets", file), "utf8"), context);
  const studio = context.window.dash_clientside.studio;
  studio.mount = () => ({id: "persistent-view", root: {}});
  studio.commit = () => {};
  return {studio, edges: context.window.turbomocMeshEdges,
    inspection: context.window.turbomocInspection};
}

const triangle = {points: [0,0,0, 1,0,0, 0,1,0], polys: [3,0,1,2]};
function meshData() {
  return {key: "volume-1", rows: [{id: "stator_1", job: "job-1", session: "saved-session",
    label: "Stator", status: "Outdated", num_blades: 20,
    surfaces: {blade: triangle}, placement: {streamwise_offset_mm: 5, angular_phase_deg: 3}}]};
}
function selection(overrides = {}) {
  return {row_id: "stator_1", job: "job-1", key: "selection-1", mode: "worst",
    metric: "skewness", context: [], note: "One original cell",
    surfaces: {inspection: {...triangle, cell_values: [.9], cell_ids: [73], color_values: [.9]}},
    ...overrides};
}
const preferences = {stator_1: {copies: 2, surfaces: ["blade"], edges: true}};
const visible = scene => scene.actors.filter(actor => actor.props.actor.visibility);
const plain = value => JSON.parse(JSON.stringify(value));

function singleRequest(inspection, ...args) {
  const batch = inspection.request(...args);
  return batch && batch.rows[0];
}

test("triangulation preserves each parent-cell value, ID and flat RGB color", () => {
  const {edges} = browser();
  const source = {points: [0,0,0, 1,0,0, 1,1,0, 0,1,0, 2,0,0],
    polys: [4,0,1,2,3, 3,1,4,2], cell_values: [.1,.9], cell_ids: [41,99], color_values: [0,1]};
  const before = JSON.stringify(source);
  const prepared = edges.prepare(source);
  assert.deepEqual([...prepared.cellValues], [.1,.1,.9]);
  assert.deepEqual([...prepared.cellIds], [41,41,99]);
  assert.deepEqual([...prepared.colors], [35,155,86, 35,155,86, 192,57,43]);
  assert.equal(JSON.stringify(source), before, "render preparation cannot mutate stored mesh geometry");
  assert.equal(edges.prepare(source), prepared, "reuse the same prepared dataset");
  assert.deepEqual([...edges.qualityColor(.5)], [244,208,63]);
});

test("polygon sections preserve metadata through all derived triangles", () => {
  const {edges} = browser();
  const prepared = edges.prepare({points: [0,0,0, 1,0,0, 2,1,0, 1,2,0, 0,1,0],
    polys: [5,0,1,2,3,4], cell_values: [.7], cell_ids: [12345678901], color_values: [.7]});
  assert.deepEqual([...prepared.cellIds], [12345678901,12345678901,12345678901]);
  assert.equal(prepared.polys.length, 12);
  assert.throws(() => edges.prepare({...triangle, cell_values: []}), /one value per face/);
});

test("quality selection retains camera identity and scalar coloring after edge-width changes", () => {
  const {studio} = browser();
  const data = meshData();
  studio.render(data, preferences, "mesh", ["stator_1"], .35);
  const original = studio.scenes.mesh;
  original.fitted = true;
  original.viewProps.cameraMarker = "user camera";
  assert.equal(visible(original).length, 2);
  studio.render(data, preferences, "mesh", ["stator_1"], .35, selection());
  const scene = studio.scenes.mesh;
  const actors = visible(scene);
  assert.equal(actors.length, 2, "inspection respects the row's displayed blade count");
  assert.equal(scene.id, original.id);
  assert.equal(scene.viewProps, original.viewProps);
  assert.equal(scene.dataKey, original.dataKey);
  assert.equal(scene.fitted, true);
  assert.deepEqual([...actors[0].props.actor.position], [0,0,5]);
  assert.deepEqual([...actors[0].props.actor.orientation], [0,0,183]);
  const cells = actors[0].props.children[0].props.children.find(item => item.type === "CellData");
  assert.deepEqual([...cells.props.children.find(item => item.props.name === "OriginalCellId").props.values], [73]);
  scene.ready = true;
  studio.render(data, preferences, "mesh", ["stator_1"], .8, selection());
  assert.equal(studio.scenes.mesh, scene, "a width change reuses actors and geometry");
  assert.equal(actors[0].props.mapper.scalarVisibility, true);
  assert.equal(actors[0].props.mapper.colorMode, 2);
  assert.equal(actors[0].props.mapper.colorByArrayName, "MeshQualityColors");
  assert.ok(actors[0].props.mapper.viewSpecificProperties.OpenGL.ShaderReplacements.length);
});

test("empty, stale and hidden selections never leave old colored cells visible", () => {
  const {studio} = browser();
  const data = meshData();
  studio.render(data, preferences, "mesh", ["stator_1"], .35, selection());
  const note = "No cells meet this threshold";
  const result = studio.render(data, preferences, "mesh", ["stator_1"], .35,
    selection({key: "empty", note, surfaces: {inspection: {points: [], polys: [],
      cell_values: [], cell_ids: [], color_values: []}}}));
  assert.equal(visible(studio.scenes.mesh).length, 0);
  assert.equal(result[1], note);
  studio.render(data, preferences, "mesh", ["stator_1"], .35, selection({job: "old-job"}));
  assert.equal(visible(studio.scenes.mesh).length, 2);
  assert.ok(visible(studio.scenes.mesh).every(actor => !actor.props.mapper.scalarVisibility));
  studio.render(data, preferences, "mesh", [], .35, selection());
  assert.equal(visible(studio.scenes.mesh).length, 0);
});

test("inspection context is neutral, translucent and optional without overwriting preferences", () => {
  const {studio} = browser();
  const data = meshData();
  const before = JSON.stringify(preferences);
  studio.render(data, preferences, "mesh", ["stator_1"], .35, selection({context: ["blade"]}));
  const actors = visible(studio.scenes.mesh);
  assert.equal(actors.length, 4);
  const context = actors.find(actor => !actor.props.mapper.scalarVisibility);
  assert.equal(context.props.property.opacity, .18);
  assert.equal(context.displayEdges, false);
  studio.render(data, preferences, "mesh", ["stator_1"], .35);
  assert.equal(visible(studio.scenes.mesh).length, 2);
  assert.ok(visible(studio.scenes.mesh).every(actor => actor.props.property.opacity === 1));
  assert.equal(JSON.stringify(preferences), before);
});

test("inspection context has its own selected boundary subset without changing camera or cell colors", () => {
  const {studio} = browser();
  const data = meshData();
  data.rows[0].surfaces = Object.fromEntries(
    ["blade", "hub", "shroud", "inlet", "outlet", "periodic_left", "periodic_right"].map(name => [name, triangle]));
  const prefs = {stator_1: {copies: 5, edges: true, surfaces: ["hub", "shroud", "inlet"]}};
  const initialData = JSON.stringify(data);
  const inspect = selection({context: ["hub", "shroud", "inlet"]});
  studio.render(data, prefs, "mesh", ["stator_1"], .35, inspect);
  const first = studio.scenes.mesh;
  first.fitted = true;
  first.viewProps.cameraMarker = "retain user camera";
  const actors = visible(first);
  assert.equal(actors.length, 20, "quality cells and each selected boundary repeat together");
  const boundaries = actors.filter(actor => !actor.props.mapper.scalarVisibility);
  assert.deepEqual([...boundaries].map(actor => actor.props.id).sort(), Array.from({length:5}, (_,i) =>
    ["hub", "inlet", "shroud"].map(name => `mesh-actor-stator_1-${i}-${name}`)).flat().sort());
  assert.ok(boundaries.every(actor => actor.props.property.opacity === .18 && !actor.displayEdges));
  assert.ok(boundaries.every(actor => JSON.stringify(actor.props.property.color) === JSON.stringify([.56,.61,.65])));
  const cellColors = actor => actor.props.children[0].props.children.find(field => field.type === "CellData")
    .props.children.find(field => field.props.name === "MeshQualityColors").props.values;
  const beforeColors = [...cellColors(actors.find(actor => actor.props.mapper.scalarVisibility))];

  prefs.stator_1.surfaces = ["blade"];
  inspect.context = ["outlet"];
  studio.render(data, prefs, "mesh", ["stator_1"], .35, inspect);
  const changed = studio.scenes.mesh;
  assert.equal(changed.viewProps, first.viewProps);
  assert.equal(changed.dataKey, first.dataKey);
  assert.equal(changed.fitted, true);
  assert.deepEqual([...visible(changed)].map(actor => actor.props.id).sort(), Array.from({length:5}, (_,i) =>
    ["inspection", "outlet"].map(name => `mesh-actor-stator_1-${i}-${name}`)).flat().sort());
  assert.deepEqual([...cellColors(visible(changed).find(actor => actor.props.mapper.scalarVisibility))], beforeColors);

  inspect.context = [];
  studio.render(data, prefs, "mesh", ["stator_1"], .35, inspect);
  assert.equal(visible(studio.scenes.mesh).length, 5, "an empty boundary selection cannot force a blade context");
  assert.equal(visible(studio.scenes.mesh)[0].props.mapper.scalarVisibility, true);
  prefs.stator_1.surfaces = ["hub", "inlet"];
  studio.render(data, prefs, "mesh", ["stator_1"], .35, selection({context: []}));
  assert.equal(visible(studio.scenes.mesh).length, 5, "the context toggle hides every selected boundary");
  studio.render(data, prefs, "mesh", ["stator_1"], .35);
  assert.equal(visible(studio.scenes.mesh).length, 10, "normal view restores all requested display copies");
  assert.ok(visible(studio.scenes.mesh).every(actor => actor.props.property.opacity === 1));
  assert.equal(JSON.stringify(data), initialData, "context controls cannot change saved scientific geometry");
});

test("inspection requests use the selected saved row and only applicable filter options", () => {
  const {inspection} = browser();
  const data = meshData();
  data.rows.push({...data.rows[0], id: "rotor_1", job: "rotor-job", session: "rotor-session", label: "Rotor"});
  const prefs = {
    stator_1: {mode: "filter", method: "threshold", threshold: .6},
    rotor_1: {mode: "filter", method: "worst", metric: "orthogonality", n: 37,
      threshold: .2, layer: 8, aspect_max: 500, context: ["blade", "hub"],
      job: "unsaved-job", session: "unsaved-session"}
  };
  const before = JSON.stringify({prefs, data});
  let request = singleRequest(inspection, prefs, "rotor_1", data, ["stator_1", "rotor_1"]);
  assert.equal(request.row_id, "rotor_1");
  assert.equal(request.job, "rotor-job");
  assert.deepEqual(plain(request.reference), {session: "rotor-session", job: "rotor-job"});
  assert.equal(request.status, "Outdated", "inspection identifies the displayed artifact's status");
  assert.deepEqual(plain(request.context), ["blade", "hub"]);
  assert.deepEqual(plain(request.options), {mode: "worst", metric: "orthogonality", n: 37});
  assert.equal(JSON.stringify({prefs, data}), before, "requests cannot mutate saved/display settings");

  prefs.rotor_1.method = "threshold";
  request = singleRequest(inspection, prefs, "rotor_1", data, ["rotor_1"]);
  assert.deepEqual(plain(request.options), {mode: "threshold", metric: "orthogonality", threshold: .2});
  prefs.rotor_1.mode = "layer";
  request = singleRequest(inspection, prefs, "rotor_1", data, ["rotor_1"]);
  assert.deepEqual(plain(request.options), {mode: "layer", metric: "orthogonality", layer: 8});
  prefs.rotor_1.metric = "aspect_ratio";
  request = singleRequest(inspection, prefs, "rotor_1", data, ["rotor_1"]);
  assert.deepEqual(plain(request.options), {mode: "layer", metric: "aspect_ratio", layer: 8});
  const previousKey = request.key;
  prefs.rotor_1.aspect_max = 10;
  request = singleRequest(inspection, prefs, "rotor_1", data, ["rotor_1"]);
  assert.equal(request.key, previousKey, "removed aspect-scale settings cannot influence requests");
  prefs.rotor_1.context = [];
  assert.deepEqual(plain(singleRequest(inspection, prefs, "rotor_1", data, ["rotor_1"]).context), []);
  prefs.rotor_1.context = ["blade"];
  assert.deepEqual(plain(singleRequest(inspection, prefs, "rotor_1", data, ["rotor_1"]).context), ["blade"]);
});

test("surface mode disables inspection and unavailable or hidden rows give explicit messages", () => {
  const {inspection} = browser();
  const data = meshData();
  assert.equal(singleRequest(inspection, {}, "stator_1", data, ["stator_1"]), null);
  assert.equal(singleRequest(inspection, {stator_1: {mode: "surface"}}, "stator_1", data, ["stator_1"]), null);
  assert.equal(inspection.active(selection(), null), null);
  const prefs = {stator_1: {mode: "filter", method: "worst", n: 5}};
  const hidden = singleRequest(inspection, prefs, "stator_1", data, []);
  assert.equal(hidden, null);
  assert.equal(inspection.active(selection(), hidden), null);
  const missing = singleRequest(inspection, prefs, "stator_1", {rows: []}, ["stator_1"]);
  assert.match(missing.error, /Generate a 3D mesh/);
  assert.equal(inspection.active(selection(), missing), null);
});

test("pending and failed inspection requests suppress previous colored cells", () => {
  const {inspection, studio} = browser();
  const data = meshData();
  const prefs = {stator_1: {mode: "filter", method: "worst", n: 5, context: []}};
  const first = singleRequest(inspection, prefs, "stator_1", data, ["stator_1"]);
  const firstResult = selection({key: first.key});
  assert.equal(inspection.active(firstResult, first), firstResult);
  studio.render(data, preferences, "mesh", ["stator_1"], .35, firstResult);
  assert.equal(visible(studio.scenes.mesh).length, 2);

  prefs.stator_1.n = 10;
  const next = singleRequest(inspection, prefs, "stator_1", data, ["stator_1"]);
  const pending = inspection.active(firstResult, next);
  assert.equal(pending.row_id, "stator_1");
  assert.equal(pending.job, "job-1");
  assert.equal(pending.mode, "pending");
  assert.equal(pending.key, next.key + "|pending");
  assert.deepEqual(plain(pending.surfaces.inspection), {
    points: [], polys: [], cell_values: [], cell_ids: [], color_values: []});
  studio.render(data, preferences, "mesh", ["stator_1"], .35, pending);
  assert.equal(visible(studio.scenes.mesh).length, 0);

  const error = {key: next.key, error: "Invalid threshold", note: "Threshold is outside its range."};
  const failed = inspection.active(error, next);
  assert.equal(failed.note, error.note);
  assert.equal(failed.surfaces.inspection.polys.length, 0);
  studio.render(data, preferences, "mesh", ["stator_1"], .35, failed);
  assert.equal(visible(studio.scenes.mesh).length, 0);
});

test("a late inspection from an older saved mesh cannot color the replacement mesh", () => {
  const {inspection, studio} = browser();
  const oldData = meshData();
  const prefs = {stator_1: {mode: "filter", method: "worst", n: 5, context: []}};
  const oldRequest = singleRequest(inspection, prefs, "stator_1", oldData, ["stator_1"]);
  const oldResult = selection({key: oldRequest.key});
  const newData = meshData();
  newData.key = "volume-2";
  newData.rows[0].job = "job-2";
  const newRequest = singleRequest(inspection, prefs, "stator_1", newData, ["stator_1"]);
  assert.notEqual(newRequest.key, oldRequest.key);
  const pending = inspection.active(oldResult, newRequest);
  assert.equal(pending.job, "job-2");
  assert.equal(pending.surfaces.inspection.polys.length, 0);
  studio.render(newData, preferences, "mesh", ["stator_1"], .35, pending);
  assert.equal(visible(studio.scenes.mesh).length, 0);
  studio.render(newData, preferences, "mesh", ["stator_1"], .35, oldResult);
  assert.ok(visible(studio.scenes.mesh).every(actor => !actor.props.mapper.scalarVisibility),
    "the renderer also checks artifact identity if called directly with a late result");
});


test("selected rows use shared controls and align nonuniform span layers by position", () => {
  const {inspection, studio} = browser();
  const data = meshData();
  data.rows[0].span_stations = [0, .1, .4, 1];
  data.rows.push({...data.rows[0], id: "rotor_1", job: "rotor-job", span_stations: [0, .2, .3, .6, 1]});
  const prefs = {stator_1: {mode: "layer", metric: "skewness", layer: 2, context: ["blade"]}};
  const request = inspection.request(prefs, ["all", "stator_1", "rotor_1"], data, ["stator_1", "rotor_1"]);
  assert.deepEqual(plain(request.rows.map(row => row.options.layer)), [2, 2]);
  prefs.stator_1.layer = 3;
  const next = inspection.request(prefs, ["all", "stator_1", "rotor_1"], data, ["stator_1", "rotor_1"]);
  assert.deepEqual(plain(next.rows.map(row => row.options.layer)), [3, 4]);
  assert.notEqual(next.key, request.key);
  const result = {key: request.key, rows: request.rows.map(row => selection({...row, mode: "layer"}))};
  studio.render(data, preferences, "mesh", ["stator_1", "rotor_1"], .35, inspection.active(result, request));
  assert.equal(visible(studio.scenes.mesh).filter(actor => actor.props.mapper.scalarVisibility).length, 3);
  const pending = inspection.active(result, next);
  studio.render(data, preferences, "mesh", ["stator_1", "rotor_1"], .35, pending);
  assert.equal(visible(studio.scenes.mesh).filter(actor => actor.props.mapper.scalarVisibility).length, 0);
  assert.equal(visible(studio.scenes.mesh).length, 3, "pending requests retain both rows' displayed context copies");
});

test("annular display uses one right-handed frame for boundaries and inspected cells", () => {
  const {studio} = browser();
  const data = meshData();
  const source = JSON.stringify(data);
  for (const family of ["axial", "radial"]) {
    const displayed = {...data, rows: data.rows.map(row => ({...row, geometry_type: family}))};
    studio.render(displayed, preferences, "mesh", ["stator_1"], .35, selection({context: ["blade"]}));
    const actors = visible(studio.scenes.mesh);
    assert.equal(actors.length, 4);
    assert.deepEqual(plain(actors.map(actor => actor.props.actor.orientation)),
      [[0,0,183],[0,0,183],[0,0,165],[0,0,165]]);
    assert.ok(actors.every(actor => JSON.stringify(actor.props.actor.position) === "[0,0,5]"));
  }
  assert.equal(JSON.stringify(data), source, "viewing-frame changes cannot change saved mesh coordinates");
  // Rz(pi) maps the old positive rotor tangent -Y onto display +Y.
  // Its two sign changes preserve cross(X,Y)=Z, unlike relabeling Y alone.
  const transformedX = [-1,0,0], transformedY = [0,-1,0];
  assert.equal(transformedX[0]*transformedY[1]-transformedX[1]*transformedY[0], 1);
});

test("machine-specific home views survive filtering and use the saved artifact family", () => {
  const {studio} = browser();
  const data = meshData();
  data.rows[0].geometry_type = "axial";
  studio.render(data, preferences, "mesh", ["stator_1"], .35);
  const axial = studio.scenes.mesh;
  assert.deepEqual(plain(axial.home), {family: "axial", direction: [-1,.12,.2], up: [0,1,0]});
  axial.fitted = true;
  studio.render(data, preferences, "mesh", ["stator_1"], .35, selection());
  assert.deepEqual(plain(studio.scenes.mesh.home), plain(axial.home));
  assert.equal(studio.scenes.mesh.viewProps, axial.viewProps);
  assert.equal(studio.scenes.mesh.fitted, true);
  data.rows[0].geometry_type = "radial";
  studio.render(data, preferences, "mesh", ["stator_1"], .35);
  assert.deepEqual(plain(studio.scenes.mesh.home), {family: "radial", direction: [0,0,1], up: [0,1,0]});
  assert.equal(studio.scenes.mesh.fitted, false, "a family change needs a fresh initial framing");
});

test("linear cascades retain their physical axes and display-copy translations", () => {
  const {studio} = browser();
  const data = meshData();
  data.rows[0] = {...data.rows[0], is_linear: true, geometry_type: "linear", pitch_mm: 8};
  studio.render(data, preferences, "geometry", ["stator_1"], .35);
  const scene = studio.scenes.geometry;
  assert.deepEqual(plain(scene.home), {family: "linear", direction: [1,1,1], up: [0,0,1]});
  const actors = visible(scene);
  assert.ok(actors.every(actor => actor.props.actor.orientation.every(value => value === 0)));
  assert.deepEqual(plain(actors.map(actor => actor.props.actor.position)), [[0,0,0],[0,8,0]]);
});
