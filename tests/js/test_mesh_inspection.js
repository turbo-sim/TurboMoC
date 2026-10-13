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
    metric: "skewness", context: false, note: "One original cell",
    surfaces: {inspection: {...triangle, cell_values: [.9], cell_ids: [73], color_values: [.9]}},
    ...overrides};
}
const preferences = {stator_1: {copies: 2, surfaces: ["blade"], edges: true}};
const visible = scene => scene.actors.filter(actor => actor.props.actor.visibility);
const plain = value => JSON.parse(JSON.stringify(value));

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
  assert.equal(actors.length, 1, "inspection displays one passage");
  assert.equal(scene.id, original.id);
  assert.equal(scene.viewProps, original.viewProps);
  assert.equal(scene.dataKey, original.dataKey);
  assert.equal(scene.fitted, true);
  assert.deepEqual([...actors[0].props.actor.position], [0,0,5]);
  assert.deepEqual([...actors[0].props.actor.orientation], [0,0,3]);
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
  studio.render(data, preferences, "mesh", ["stator_1"], .35, selection({context: true}));
  const actors = visible(studio.scenes.mesh);
  assert.equal(actors.length, 2);
  const context = actors.find(actor => !actor.props.mapper.scalarVisibility);
  assert.equal(context.props.property.opacity, .18);
  assert.equal(context.displayEdges, false);
  studio.render(data, preferences, "mesh", ["stator_1"], .35);
  assert.equal(visible(studio.scenes.mesh).length, 2);
  assert.ok(visible(studio.scenes.mesh).every(actor => actor.props.property.opacity === 1));
  assert.equal(JSON.stringify(preferences), before);
});

test("inspection context follows the selected boundary subset without changing camera or cell colors", () => {
  const {studio} = browser();
  const data = meshData();
  data.rows[0].surfaces = Object.fromEntries(
    ["blade", "hub", "shroud", "inlet", "outlet", "periodic_left", "periodic_right"].map(name => [name, triangle]));
  const prefs = {stator_1: {copies: 5, edges: true, surfaces: ["hub", "shroud", "inlet"]}};
  const initialData = JSON.stringify(data);
  const inspect = selection({context: true});
  studio.render(data, prefs, "mesh", ["stator_1"], .35, inspect);
  const first = studio.scenes.mesh;
  first.fitted = true;
  first.viewProps.cameraMarker = "retain user camera";
  const actors = visible(first);
  assert.equal(actors.length, 4, "one quality subset and one copy of each selected boundary");
  const boundaries = actors.filter(actor => !actor.props.mapper.scalarVisibility);
  assert.deepEqual([...boundaries].map(actor => actor.props.id).sort(), [
    "mesh-actor-stator_1-0-hub", "mesh-actor-stator_1-0-inlet", "mesh-actor-stator_1-0-shroud"]);
  assert.ok(boundaries.every(actor => actor.props.property.opacity === .18 && !actor.displayEdges));
  assert.ok(boundaries.every(actor => JSON.stringify(actor.props.property.color) === JSON.stringify([.56,.61,.65])));
  const cellColors = actor => actor.props.children[0].props.children.find(field => field.type === "CellData")
    .props.children.find(field => field.props.name === "MeshQualityColors").props.values;
  const beforeColors = [...cellColors(actors.find(actor => actor.props.mapper.scalarVisibility))];

  prefs.stator_1.surfaces = ["outlet"];
  studio.render(data, prefs, "mesh", ["stator_1"], .35, inspect);
  const changed = studio.scenes.mesh;
  assert.equal(changed.viewProps, first.viewProps);
  assert.equal(changed.dataKey, first.dataKey);
  assert.equal(changed.fitted, true);
  assert.deepEqual([...visible(changed)].map(actor => actor.props.id).sort(), [
    "mesh-actor-stator_1-0-inspection", "mesh-actor-stator_1-0-outlet"]);
  assert.deepEqual([...cellColors(visible(changed).find(actor => actor.props.mapper.scalarVisibility))], beforeColors);

  prefs.stator_1.surfaces = [];
  studio.render(data, prefs, "mesh", ["stator_1"], .35, inspect);
  assert.equal(visible(studio.scenes.mesh).length, 1, "an empty boundary selection cannot force a blade context");
  assert.equal(visible(studio.scenes.mesh)[0].props.mapper.scalarVisibility, true);
  prefs.stator_1.surfaces = ["hub", "inlet"];
  studio.render(data, prefs, "mesh", ["stator_1"], .35, selection({context: false}));
  assert.equal(visible(studio.scenes.mesh).length, 1, "the context toggle hides every selected boundary");
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
      threshold: .2, layer: 8, aspect_max: 500, context: ["surfaces"],
      job: "unsaved-job", session: "unsaved-session"}
  };
  const before = JSON.stringify({prefs, data});
  let request = inspection.request(prefs, "rotor_1", data, ["stator_1", "rotor_1"]);
  assert.equal(request.row_id, "rotor_1");
  assert.equal(request.job, "rotor-job");
  assert.deepEqual(plain(request.reference), {session: "rotor-session", job: "rotor-job"});
  assert.equal(request.status, "Outdated", "inspection identifies the displayed artifact's status");
  assert.equal(request.context, true);
  assert.deepEqual(plain(request.options), {mode: "worst", metric: "orthogonality", n: 37});
  assert.equal(JSON.stringify({prefs, data}), before, "requests cannot mutate saved/display settings");

  prefs.rotor_1.method = "threshold";
  request = inspection.request(prefs, "rotor_1", data, ["rotor_1"]);
  assert.deepEqual(plain(request.options), {mode: "threshold", metric: "orthogonality", threshold: .2});
  prefs.rotor_1.mode = "layer";
  request = inspection.request(prefs, "rotor_1", data, ["rotor_1"]);
  assert.deepEqual(plain(request.options), {mode: "layer", metric: "orthogonality", layer: 8});
  prefs.rotor_1.metric = "aspect_ratio";
  request = inspection.request(prefs, "rotor_1", data, ["rotor_1"]);
  assert.deepEqual(plain(request.options), {mode: "layer", metric: "aspect_ratio", layer: 8});
  const previousKey = request.key;
  prefs.rotor_1.aspect_max = 10;
  request = inspection.request(prefs, "rotor_1", data, ["rotor_1"]);
  assert.equal(request.key, previousKey, "removed aspect-scale settings cannot influence requests");
  prefs.rotor_1.context = [];
  assert.equal(inspection.request(prefs, "rotor_1", data, ["rotor_1"]).context, false);
  prefs.rotor_1.context = ["blade"];
  assert.equal(inspection.request(prefs, "rotor_1", data, ["rotor_1"]).context, true,
    "existing browser preferences can still enable boundary context");
});

test("surface mode disables inspection and unavailable or hidden rows give explicit messages", () => {
  const {inspection} = browser();
  const data = meshData();
  assert.equal(inspection.request({}, "stator_1", data, ["stator_1"]), null);
  assert.equal(inspection.request({stator_1: {mode: "surface"}}, "stator_1", data, ["stator_1"]), null);
  assert.equal(inspection.active(selection(), null), null);
  const prefs = {stator_1: {mode: "filter", method: "worst", n: 5}};
  const hidden = inspection.request(prefs, "stator_1", data, []);
  assert.match(hidden.error, /hidden.*Show/);
  assert.equal(inspection.active(selection(), hidden), null);
  const missing = inspection.request(prefs, "stator_1", {rows: []}, ["stator_1"]);
  assert.match(missing.error, /Generate a 3D mesh/);
  assert.equal(inspection.active(selection(), missing), null);
});

test("pending and failed inspection requests suppress previous colored cells", () => {
  const {inspection, studio} = browser();
  const data = meshData();
  const prefs = {stator_1: {mode: "filter", method: "worst", n: 5, context: []}};
  const first = inspection.request(prefs, "stator_1", data, ["stator_1"]);
  const firstResult = selection({key: first.key});
  assert.equal(inspection.active(firstResult, first), firstResult);
  studio.render(data, preferences, "mesh", ["stator_1"], .35, firstResult);
  assert.equal(visible(studio.scenes.mesh).length, 1);

  prefs.stator_1.n = 10;
  const next = inspection.request(prefs, "stator_1", data, ["stator_1"]);
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
  const oldRequest = inspection.request(prefs, "stator_1", oldData, ["stator_1"]);
  const oldResult = selection({key: oldRequest.key});
  const newData = meshData();
  newData.key = "volume-2";
  newData.rows[0].job = "job-2";
  const newRequest = inspection.request(prefs, "stator_1", newData, ["stator_1"]);
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
