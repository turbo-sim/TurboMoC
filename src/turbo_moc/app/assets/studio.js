
/* Browser-only display copies and widths. CFD coordinates remain immutable. */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  studio: {
    scenes: {},
    roots: {},
    home: function(data) {
      // Use the displayed artifact's family, including when its design is old.
      // Row visibility must not change the shared camera convention.
      const row = (data.rows || [data])[0] || {};
      const family = row.geometry_type || (row.is_linear ? "linear" : "axial");
      if (row.home) return {family, direction: row.home.direction, up: row.home.up};
      if (family === "radial") return {family, direction: [0,0,1], up: [0,1,0]};
      if (family === "linear") return {family, direction: [1,1,1], up: [0,0,1]};
      return {family, direction: [-1,.12,.20], up: [0,1,0]};
    },
    element: function(description) {
      const props = Object.assign({}, description.props);
      const children = props.children || [];
      delete props.children;
      // These are the installed dash-vtk React components, with the same
      // bundled VTK/shader implementation as the original Dash viewer.
      props.key = props.id || props.name || description.type;
      return React.createElement(window.dash_vtk[description.type], props,
        ...children.map(child => window.dash_clientside.studio.element(child)));
    },
    mount: function(pane) {
      const studio = window.dash_clientside.studio;
      const target = document.getElementById(pane + "-scene");
      let mounted = studio.roots[pane];
      if (!mounted) {
        const host = document.createElement("div");
        host.style.height = "100%";
        host.style.width = "100%";
        target.appendChild(host);
        // Dash remounts descendants when a serialized children property is
        // replaced, even with stable IDs. Own this small React subtree so
        // camera/context lifetime follows React reconciliation instead.
        const root = ReactDOM.createRoot ? ReactDOM.createRoot(host)
          : {render: element => ReactDOM.render(element, host)};
        const navigation = window.turbomocOrientation.create(target, pane);
        mounted = {host, root, navigation, bindView: instance => {
          mounted.view = instance;
          navigation.attach(instance);
        },
          id: pane + "-vtk-" + Date.now()};
        studio.roots[pane] = mounted;
      } else if (mounted.host.parentElement !== target) {
        target.appendChild(mounted.host);
        target.appendChild(mounted.navigation.host);
      }
      return mounted;
    },
    commit: function(scene) {
      const studio = window.dash_clientside.studio;
      if (studio.scenes[scene.pane] !== scene) return;
      studio.roots[scene.pane].navigation.setHome(scene.home.direction, scene.home.up);
      studio.roots[scene.pane].navigation.setEnabled(scene.active);
      scene.root.render(studio.view(scene));
    },
    updateMeshStyle: function(scene) {
      const studio = window.dash_clientside.studio;
      if (!scene.ready || !scene.active || studio.scenes[scene.pane] !== scene) return;
      const element = document.getElementById(scene.id);
      const canvas = element && element.querySelector("canvas");
      const size = canvas && canvas.getBoundingClientRect();
      const pixelRatio = size && size.width > 0 ? canvas.width / size.width : 1;
      const styleKey = [scene.width, scene.edges, pixelRatio].join("|");
      if (scene.styleKey === styleKey) return;
      scene.styleKey = styleKey;
      for (const actor of scene.actors) {
        if (!actor.props.actor.visibility) continue;
        actor.props.mapper = window.turbomocMeshEdges.mapper(
          scene.width, actor.displayEdges, pixelRatio, actor.scalarMapper);
      }
      studio.commit(scene);
      // Wait for React to apply mapper settings before requesting a new frame.
      requestAnimationFrame(() => {
        if (studio.scenes[scene.pane] === scene && scene.active) {
          scene.viewProps.triggerRender = Date.now();
          studio.commit(scene);
        }
      });
    },
    fit: function(id, key, pane, count, attempt) {
      const studio = window.dash_clientside.studio;
      const scene = studio.scenes[pane];
      if (!scene || !scene.active || scene.key !== key) return;
      const el = document.getElementById(id);
      if (el && el.querySelectorAll("[id^='" + pane + "-poly-']").length === count) {
        if (pane === "mesh") {
          // dash-vtk mounts point arrays before applying the parent's mapper
          // options. Its bundled vtk.js omits custom attributes from the VBO
          // cache key. Refresh the edge arrays once after mount, then enable
          // the shader only when their GPU buffers have been rebuilt.
          for (const actor of scene.actors) {
            if (!actor.props.actor.visibility) continue;
            const pointData = actor.props.children[0].props.children.find(field => field.type === "PointData");
            const array = pointData && pointData.props.children.find(field =>
              field.props.name === "MeshEdgeCoordinates");
            if (array) array.props.values = array.props.values.slice();
          }
          studio.commit(scene);
          requestAnimationFrame(() => {
            if (studio.scenes[pane] !== scene || !scene.active) return;
            scene.ready = true;
            studio.updateMeshStyle(scene);
          });
        }
        // A dataset can finish while its workspace is hidden. Fit on the first
        // visible resize, then preserve the user's camera on later resizes.
        const observer = new ResizeObserver(entries => {
          if (studio.scenes[pane] !== scene || !scene.active) return;
          const size = entries[0].contentRect;
          if (pane === "mesh" && size.width > 0 && size.height > 0)
            requestAnimationFrame(() => studio.updateMeshStyle(scene));
          if (size.width > 0 && size.height > 0 && !scene.fitted) {
            scene.fitted = true;
            setTimeout(() => {
              if (studio.scenes[pane] === scene && scene.active && document.getElementById(id)) {
                scene.viewProps.triggerResetCamera = Date.now();
                scene.viewProps.triggerRender = Date.now();
                studio.commit(scene);
              }
            }, 350);
          }
        });
        scene.observer = observer;
        observer.observe(el);
      } else if (attempt < 100) {
        setTimeout(() => studio.fit(id, key, pane, count, attempt + 1), 50);
      }
    },
    view: function(scene) {
      const studio = window.dash_clientside.studio;
      return window.turbomocOrientation.vtkView(
        Object.assign({}, scene.viewProps, {id: scene.id,
          background: [.98,.985,.987], style: {width: "100%", height: "100%"},
          cameraPosition: scene.home.direction, cameraViewUp: scene.home.up}),
        scene.actors.map(actor => studio.element(actor)), studio.roots[scene.pane].bindView);
    },
    hideActors: function(actors) {
      return actors.map(actor => Object.assign({}, actor, {props: Object.assign({}, actor.props, {
        actor: Object.assign({}, actor.props.actor, {visibility: false})
      })}));
    },
    render: function(data, settings, pane, plottedRows, width, inspection) {
      const studio = window.dash_clientside.studio;
      const previous = studio.scenes[pane];
      settings = settings || {};
      const rows = data && (data.rows || [data]).filter(row =>
        plottedRows == null || plottedRows.includes(row.id || 'stator_1'));
      if (!rows || !rows.length) {
        if (previous && previous.observer) previous.observer.disconnect();
        const note = plottedRows && !plottedRows.length ? "Select a row under Configure display > Blade rows to display it."
                   : data && data.rows && data.rows.length ? "No generated mesh or geometry for the shown rows."
                   : pane === "mesh" ? "Generate a current 3D mesh to show its boundary cells."
                                    : "Compute current MoC to show geometry.";
        if (!previous) return [window.dash_clientside.no_update, note];
        const hidden = Object.assign({}, previous, {key: "empty", active: false,
          ready: false, observer: null, actors: studio.hideActors(previous.actors)});
        studio.scenes[pane] = hidden;
        // Retain the View even while no current artifact is selected. Its
        // bundled timers and camera listeners then always reference live VTK
        // objects; stale data are hidden rather than drawn as current results.
        studio.commit(hidden);
        return [window.dash_clientside.no_update, note];
      }
      // Selection data belong to exactly one immutable mesh artifact. Never
      // render a late result on a replacement mesh or a hidden blade row.
      const selections = pane === "mesh" && inspection ? (inspection.rows || [inspection]) : [];
      const home = studio.home(data);
      const selectionFor = row => selections.find(item => item.mode !== "off" &&
        row.id === item.row_id && row.job === item.job);
      const inspecting = rows.some(row => selectionFor(row));
      const lineWidth = Number(width) > 0 ? Number(width) : 0.35;
      // Visibility changes actor selection, but retains the numerical data key
      // and camera. No regeneration or removal of hidden row results is needed.
      const key = [data.key, home.family, JSON.stringify(rows.map(row => [row.id, settings[row.id]])),
        inspecting ? JSON.stringify(selections.map(item => [item.row_id, item.job, item.key, item.context])) : ""].join("|");
      const note = inspecting ? selections.map(item => item.note).join(" | ") : rows.map(row => (row.label || row.id) + ": " +
        Math.min(row.is_linear ? 25 : row.num_blades, Math.max(1, Math.round((settings[row.id] || {}).copies || 1)))
        + " displayed").join(" · ") + " · XYZ [mm] · " +
        (pane === "mesh" ? "actual volume boundary cells" : "geometry display tessellation") +
        ". Drag to rotate; scroll to zoom.";
      if (previous && previous.key === key) {
        // Width and edge visibility change only the shader. Reuse the actors,
        // their data and the camera; no numerical job or geometry transfer.
        if (pane === "mesh") {
          previous.width = lineWidth;
          studio.updateMeshStyle(previous);
        }
        return [window.dash_clientside.no_update, note];
      }
      if (previous && previous.observer) previous.observer.disconnect();
      const colors = {hub: [.65,.70,.72], shroud: [.65,.70,.72],
        inlet: [201/255,79/255,79/255], outlet: [201/255,79/255,79/255], periodic_left: [77/255,153/255,102/255],
        periodic_right: [77/255,153/255,102/255]};
      const actors = [];
      // Row datasets remain independent. Translation/rotation is a display
      // transform, so changing assembly placement never mutates mesh nodes.
      for (const row of rows) {
        const rowInspection = selectionFor(row);
        const inspecting = !!rowInspection;
        const display = settings[row.id] || {};
        const count = Math.max(1, Math.round(display.copies || 1));
        const names = inspecting ? ["inspection", ...(rowInspection.context || [])]
          : display.surfaces || ["blade", "hub"];
        const rowCount = row.is_linear ? count : Math.min(row.num_blades, count);
        const placement = row.placement || {};
        const rowId = row.id || 'stator_1';
        for (let i = 0; i < rowCount; ++i) for (const name of names) {
        const quality = inspecting && name === "inspection";
        const context = inspecting && !quality;
        const original = quality ? (rowInspection.surfaces || {}).inspection : row.surfaces[name];
        if (!original || !original.polys.length) continue;
        const s = pane === "mesh" ? window.turbomocMeshEdges.prepare(original) : original;
        const arrays = [];
        if (s.normals) arrays.push({namespace: "dash_vtk", type: "DataArray", props: {
            name: "Normals", registration: "setNormals", numberOfComponents: 3,
            type: "Float32Array", values: s.normals}});
        if (s.coordinates) arrays.push({namespace: "dash_vtk", type: "DataArray", props: {
            id: pane + "-edge-coordinates-" + rowId + "-" + i + "-" + name,
            name: "MeshEdgeCoordinates", numberOfComponents: 3,
            type: "Float32Array", values: s.coordinates}});
        const fields = arrays.length ? [{namespace: "dash_vtk", type: "PointData",
          props: {children: arrays}}] : [];
        if (quality) {
          const cells = [];
          for (const [name, values, type, components] of [
            ["MeshQualityColors", s.colors, "Uint8Array", 3],
            ["MeshQualityValue", s.cellValues, "Float64Array", 1],
            ["OriginalCellId", s.cellIds, "Float64Array", 1]]) {
            if (values) cells.push({namespace: "dash_vtk", type: "DataArray", props: {
              name, values, type, numberOfComponents: components,
              registration: name === "MeshQualityColors" ? "setScalars" : "addArray"}});
          }
          fields.push({namespace: "dash_vtk", type: "CellData", props: {children: cells}});
        }
        const hex = (display.colors || {})[name];
        const fallback = name === "blade" ? (row.role === "rotor" ? [239/255,140/255,38/255]
          : [71/255,125/255,179/255]) : colors[name] || [1,1,1];
        const color = context ? [.56, .61, .65] : /^#[0-9a-f]{6}$/i.test(hex || "") ?
          [1, 3, 5].map(offset => parseInt(hex.slice(offset, offset + 2), 16) / 255) : fallback;
        // vtk.js: USE_CELL_FIELD_DATA=4, DIRECT_SCALARS=2. Explicit RGB values
        // keep the same green-to-red legend across metrics, subsets and layers.
        const scalarMapper = quality && s.colors ? {scalarVisibility: true,
          scalarMode: 4, colorMode: 2, colorByArrayName: "MeshQualityColors",
          interpolateScalarsBeforeMapping: false} : {scalarVisibility: false};
        actors.push({namespace: "dash_vtk", type: "GeometryRepresentation",
          displayEdges: !context && !!display.edges, scalarMapper, props: {
          id: pane + "-actor-" + rowId + "-" + i + "-" + name,
          actor: {visibility: true,
            // A proper rotation gives the annular display frame (-X,-Y,Z).
            // +Y follows positive rotor motion at the reference passage; the
            // source coordinates, connectivity and exported meshes stay fixed.
            orientation: [0,0,row.is_linear ? 0 : 180 + (placement.angular_phase_deg || 0) - 360 * i / row.num_blades],
            position: row.is_linear ? [0, i * row.pitch_mm, 0] : [0, 0, placement.streamwise_offset_mm || 0]},
          mapper: pane === "mesh" ? window.turbomocMeshEdges.mapper(lineWidth, false, 1, scalarMapper)
                                  : {scalarVisibility: false},
          property: {color, edgeColor: [0,0,0],
            edgeVisibility: false, ambient: quality ? 1 : .25, diffuse: quality ? 0 : .75,
            opacity: context ? .18 : 1,
            interpolation: pane === "geometry" ? 2 : 1},
          children: [{namespace: "dash_vtk", type: "PolyData", props: {
            id: pane + "-poly-" + rowId + "-" + i + "-" + name, points: s.points, polys: s.polys,
            connectivity: "manual", children: fields}}] }});
      }
      }
      // dash-vtk 0.0.9 does not cancel its mount-time camera timer or debounced
      // bounds callback on unmount. Its representation cube axes also attach
      // camera listeners. Keep one View and a stable actor pool for this pane:
      // update existing datasets in place and hide unused copies/surfaces.
      // Stable order and IDs let React retain each representation's lifetime.
      const desired = new Map(actors.map(actor => [actor.props.id, actor]));
      const pool = [];
      if (previous) {
        for (const old of previous.actors) {
          const current = desired.get(old.props.id);
          pool.push(current || studio.hideActors([old])[0]);
          desired.delete(old.props.id);
        }
      }
      pool.push(...desired.values());
      const mounted = studio.mount(pane);
      const id = mounted.id;
      const fitted = !!(previous && previous.dataKey === data.key && previous.home.family === home.family && previous.fitted);
      const scene = {id, pane, root: mounted.root, viewProps: previous ? previous.viewProps : {},
        key, dataKey: data.key, home, width: lineWidth,
        edges: JSON.stringify(rows.map(row => !!(settings[row.id] || {}).edges)),
        actors: pool, active: actors.length > 0, ready: false, fitted};
      studio.scenes[pane] = scene;
      studio.commit(scene);
      // Fit new numerical datasets; display controls preserve the camera.
      if (scene.active) setTimeout(() => studio.fit(id, key, pane, pool.length, 0), 0);
      return [window.dash_clientside.no_update,
              inspecting || actors.length ? note : "Select a surface to display."];
    }
  }
});
// Native details elements keep their open state. Notify the renderers when a
// collapsed plot reopens; data and numerical artifacts do not change.
document.addEventListener("toggle", event => {
  if (event.target.tagName === "DETAILS" && event.target.open)
    window.dispatchEvent(new Event("resize"));
}, true);

// Native color controls feed Dash stores without adding a UI dependency.
function mountSurfaceColors() {
  document.querySelectorAll('.surface-color-picker').forEach(host => {
    let picker = host.querySelector('input');
    if (!picker) {
      picker = document.createElement('input');
      picker.type = 'color';
      picker.addEventListener('input', () => {
        window.dash_clientside.set_props(JSON.parse(host.dataset.colorId), {data: picker.value});
      });
      host.appendChild(picker);
    }
    picker.setAttribute('aria-label', host.dataset.colorLabel);
    picker.title = host.dataset.colorLabel;
    if (picker.value !== host.dataset.color) picker.value = host.dataset.color;
  });
}
new MutationObserver(mountSurfaceColors).observe(document.documentElement,
  {subtree: true, childList: true, attributes: true, attributeFilter: ['data-color']});
mountSurfaceColors();

